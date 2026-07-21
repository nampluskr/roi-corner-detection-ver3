# Training and Inference Flow

이 문서는 CSV가 dataloader가 되고, wrapper가 epoch를 학습하며, checkpoint가 평가와 예측에 사용되는 전체
실행 흐름을 설명한다. 학습, validation, 독립 evaluation, prediction은 같은 model과 data contract를
공유하지만 목적과 저장 결과가 다르다.

## 1. 네 실행 단계의 차이

처음 project를 사용할 때 가장 먼저 구분할 단계는 다음 네 가지다.

| 단계 | gradient 계산 | 정답 사용 목적 | 주요 결과 |
| --- | --- | --- | --- |
| training | 사용 | loss와 parameter update | epoch별 train result |
| validation | 사용하지 않음 | loss, metric, early stopping | epoch별 valid result |
| evaluation | 사용하지 않음 | 공통 test metric | `metrics.json` |
| prediction | 사용하지 않음 | target과 prediction 비교 기록 | `predictions.csv` |

validation은 training script 안에서 매 epoch 실행된다. evaluation과 prediction은 학습이 끝난 checkpoint를
별도 script로 읽어 test split에서 실행한다.

## 2. 전체 lifecycle

일반적인 실험 순서는 다음과 같다.

```text
labeled CSV and images
-> deterministic train, valid, test split
-> train and validation epochs
-> restore best validation weights
-> save history.json
-> optionally save model.pth
-> load model.pth in evaluation
-> save metrics.json
-> load model.pth in prediction
-> save predictions.csv
```

`history.json`이 있다고 해서 checkpoint가 존재하는 것은 아니다. training script는 항상 history를
저장하지만 `model.pth`는 `--save` option을 지정한 경우에만 저장한다.

## 3. CLI configuration 생성

세 script는 모두 `scripts.config.parse_args()`를 사용한다. parser는 command line을 `argparse.Namespace`로
바꾸고 지정하지 않은 값에 공통 default를 넣는다.

주요 default는 다음과 같다.

| 항목 | 기본값 |
| --- | ---: |
| dataset stage | `public` |
| model | `reg` |
| network | `custom` |
| head | `gap` |
| image size | `224` |
| batch size | `4` |
| max epochs | `5` |
| patience | `3` |
| warmup epochs | `1` |
| seed | `42` |
| train, valid, test limit | `2000`, `1000`, `1000` |

global default인 `custom`과 `gap`은 `reg` 조합에 맞는다. 다른 model을 선택할 때는 compatible network와
head를 명시해야 한다. 예를 들어 `seg`에는 `--head mask`가 필요하다.

## 4. Dataset 생성과 split

`get_dataset()`은 여러 CSV의 행을 하나의 dataset으로 연결한 뒤 seed 기반 random split을 두 번 수행한다.

```text
all samples
-> 60% train + 40% temporary
-> temporary 50% valid + 50% test
-> final ratio 60% train, 20% valid, 20% test
```

표본 수가 `N`이면 train 수는 `int(0.6N)`이다. 남은 수를 다시 절반으로 나누므로 홀수 표본에서는 valid와
test 수가 한 개 차이날 수 있다. 같은 CSV 순서와 같은 seed를 사용하면 세 script가 같은 split을 다시
구성한다.

`--train_size`, `--valid_size`, `--test_size`는 split을 만든 뒤 각 subset에서 사용할 표본 수를 제한한다.
값이 실제 split보다 크면 slicing 결과상 사용 가능한 전체 표본만 선택된다.

## 5. Transform과 batch 생성

모든 split은 image를 `(image_size, image_size)`로 resize하고 tensor 변환과 ImageNet normalization을
적용한다. train split에는 확률적 augmentation이 더해진다.

```text
train: resize -> flips -> rotation -> color jitter -> blur -> tensor -> normalize
valid/test: resize -> tensor -> normalize
```

geometric transform은 image와 normalized corner를 함께 바꾸며 corner 순서를 유지한다. photometric
transform은 image만 바꾸고 corner는 그대로 전달한다.

dataloader가 만드는 labeled batch의 기본 shape는 다음과 같다.

| 값 | shape | 의미 |
| --- | --- | --- |
| `images` | `(B, 3, 224, 224)` | normalized RGB batch |
| `targets` | `(B, 4, 2)` | `TL`, `TR`, `BR`, `BL` normalized corner |

train dataloader는 shuffle과 `drop_last=True`를 사용한다. 마지막 batch가 batch size보다 작으면 학습에서
제외된다. validation과 test는 shuffle과 drop을 사용하지 않는다. `num_workers`를 명시하면 모든 split에
같은 값이 전달되며, current CLI 기본값은 `4`다.

## 6. Wrapper 조립

script는 `get_wrapper_kwargs()`로 `network`, `head`, `warmup_epochs`를 모아 `get_wrapper(model, ...)`에
전달한다. factory는 model key에 맞는 wrapper를 만든다.

wrapper는 다음 object를 포함한다.

- image에서 raw output을 만드는 model
- common corner를 loss target으로 바꾸는 preprocessor
- raw output을 final corner로 바꾸는 postprocessor
- parameter를 갱신하는 optimizer와 scheduler
- 학습 목적을 계산하는 loss state
- validation 진행을 확인하는 metric state

CLI의 `image_size`는 dataloader resize에는 전달되지만 현재 `get_wrapper_kwargs()`에는 포함되지 않는다.
dense target이나 pixel scale을 내부 기본값 224로 만드는 wrapper가 있으므로 standard workflow에서는
`--image_size 224`를 유지해야 한다.

## 7. Training 시작과 warmup

`Trainer.fit_early_stop()`은 첫 epoch 전에 `wrapper.on_fit_start(max_epochs)`를 호출한다. warmup을 실제로
적용하는 wrapper는 초기 phase에서 backbone parameter의 `requires_grad`를 끄고 phase 1 optimizer와
scheduler를 다시 만든다.

warmup epoch 수를 `W`라고 하면 개념적인 phase는 다음과 같다.

```text
epoch 1 ... W: frozen backbone, train task-specific layers
epoch W+1 ...: unfreeze backbone, rebuild optimizer and scheduler
```

`on_epoch_start()`는 `epoch == applied_warmup_epochs + 1`일 때 phase 2를 시작한다. CLI로 받은
`warmup_epochs`와 wrapper가 실제 적용하는 `applied_warmup_epochs`는 구분해야 한다. 예를 들어
`torchseg`는 option을 받을 수 있지만 현재 phase warmup을 적용하지 않는다.

optimizer를 다시 만들면 phase 1 optimizer의 momentum 같은 내부 state는 phase 2에 이어지지 않는다.
이는 backbone을 포함한 새 parameter group으로 학습을 시작하기 위한 current design이다.

## 8. 한 training batch의 내부 동작

기본 `BaseWrapper.train_step()`은 다음 순서로 실행된다.

```text
set model.train()
-> move images and targets to device
-> zero optimizer gradients
-> model forward
-> preprocess common corners
-> compute named losses
-> weighted loss sum
-> backward
-> optimizer step
-> postprocess raw output
-> update common metrics
```

loss는 model-specific target과 raw output을 비교한다. metric은 postprocessor가 만든 common final corner와
dataset corner를 비교한다. 따라서 같은 batch에서도 loss와 metric은 서로 다른 표현을 볼 수 있다.

named loss 값을 $L_k$, 각 loss object의 weight를 $w_k$라고 하면 backward에 쓰는 total loss는 다음과 같다.

$$
L_{total} = \sum_k w_k L_k
$$

progress bar에 표시되는 개별 loss와 metric은 stateful running result다. 마지막 batch 하나의 값이 아니라
현재 epoch 동안 `update`된 값의 누적 결과다.

## 9. Validation epoch

training epoch가 끝나면 trainer는 valid dataloader를 순회한다. `eval_step()`은 `model.eval()`과
`torch.no_grad()`를 사용하므로 parameter를 갱신하지 않는다.

```text
valid images and corners
-> raw output
-> model-specific validation losses
-> final corners
-> wrapper metrics
```

validation도 loss와 metric state를 시작 전에 reset한다. train state와 valid state가 섞이지 않으며 다음
epoch에도 이전 값이 이어지지 않는다.

## 10. Scheduler update

각 epoch의 train과 validation이 끝나면 `wrapper.on_epoch_end(valid_score)`가 scheduler를 한 번 갱신한다.
일반 scheduler는 인자 없이 `step()`을 호출하고, `ReduceLROnPlateau`는 validation score를 전달한다.

trainer log의 `lr`은 optimizer의 마지막 parameter group learning rate다. 여러 group이 서로 다른 learning
rate를 사용하면 log 숫자 하나만으로 모든 group의 값을 알 수 없다는 점에 주의한다.

## 11. Early stopping

기본 monitor는 validation `iou`이고 mode는 `max`다. 이전 best score보다 `min_delta=1e-4` 이상 높아야
개선으로 판단한다.

현재 score를 $s_t$, best score를 $s^*$라고 하면 개선 조건은 다음과 같다.

$$
s_t > s^* + 10^{-4}
$$

개선되면 model `state_dict`를 CPU tensor clone으로 memory에 보관하고 wait를 0으로 초기화한다. 개선되지
않으면 wait를 늘리며 `wait >= patience`일 때 epoch 반복을 중단한다.

monitor key가 wrapper의 validation 결과에 없으면 early stopping을 비활성화하고 남은 max epoch를 계속
실행한다. 따라서 log의 `early stopping disabled` 메시지를 확인해야 한다.

학습이 끝나면 trainer는 memory에 보관한 best state를 model에 다시 load한다. 이후 `--save`로 기록되는
checkpoint는 마지막 epoch가 아니라 best validation epoch의 weight다. `history.json`에는 중단 시점까지의
모든 epoch 기록이 남는다.

## 12. Training output

`scripts/train.py`는 `Trainer.save()`를 호출해 항상 `history.json`을 기록한다. output directory가 없으면
자동으로 생성된다. `--save`가 있으면 복원된 best model을 checkpoint로 저장한다.

```text
training output/
├── history.json
├── model.pth       # only with --save
└── run.log
```

`--checkpoint`를 training과 함께 지정하면 save 경로를 바꾼다. 이 option은 기존 checkpoint에서 학습을
재개하는 기능이 아니다. training script는 시작 시 checkpoint를 load하지 않는다.

## 13. 독립 evaluation

evaluation은 `scripts/evaluate.py`로 실행하며 `--checkpoint`가 필수다. 동작 순서는 다음과 같다.

```text
parse options
-> build test dataloader
-> build same model assembly
-> load checkpoint state_dict
-> predict final corners
-> update six common metrics
-> save metrics.json
```

evaluator는 wrapper에 들어 있는 validation metric set을 그대로 사용하지 않는다. 매 evaluation마다 다음
공통 metric을 새 instance로 만든다.

| key | 의미 | 좋은 방향 |
| --- | --- | --- |
| `iou` | predicted polygon과 target polygon의 area IoU | 높을수록 좋음 |
| `mcd` | 네 대응 corner distance의 sample mean | 낮을수록 좋음 |
| `maxcd` | 한 sample에서 가장 큰 corner distance의 dataset mean | 낮을수록 좋음 |
| `pck_002` | distance 0.02 이내 corner 비율 | 높을수록 좋음 |
| `pck_005` | distance 0.05 이내 corner 비율 | 높을수록 좋음 |
| `sr` | 모든 prediction coordinate가 finite인 sample 비율 | 높을수록 좋음 |

evaluation의 metric은 normalized coordinate에서 계산된다. image가 224라면 distance 0.02는 한 축 기준 약
4.48 pixel scale이지만 Euclidean distance이므로 단순한 x 또는 y 오차 하나와 같지는 않다.

## 14. 독립 prediction

prediction은 `scripts/predict.py`로 실행하며 마찬가지로 checkpoint가 필수다. test batch마다
`wrapper.predict_step(images)`를 호출하고 target과 prediction을 sample row로 저장한다.

```text
checkpoint + test images
-> raw output
-> postprocessor
-> final corners
-> pair with test targets
-> predictions.csv
```

현재 predictor는 labeled `CornerDataset`을 사용하므로 target column도 함께 기록한다. `index`는 원본 CSV
행 번호나 image path가 아니라 test dataloader에서 나온 순차 번호다.

`success`는 prediction의 모든 coordinate가 finite인지 확인한 값이다. zero corner나 center fallback처럼
finite하지만 부정확한 결과는 `True`일 수 있다. 따라서 success만 보지 말고 target과 prediction 좌표,
IoU와 distance를 함께 확인해야 한다.

## 15. Checkpoint assembly 일치

checkpoint는 `wrapper.model.state_dict()`만 저장한다. model, network, head, image size, optimizer,
scheduler, epoch, CLI config metadata는 포함하지 않는다.

평가할 때는 학습 시 사용한 model assembly를 command line에서 다시 지정해야 한다.

```text
training:   model=seg, network=resnet18, head=mask
evaluation: model=seg, network=resnet18, head=mask
```

다른 조합으로 object를 만든 뒤 state dictionary를 load하면 missing key, unexpected key, tensor size mismatch가
발생할 수 있다. 우연히 load되더라도 preprocessor나 postprocessor 설정이 다르면 동일한 실험이 아니다.

checkpoint는 optimizer state를 포함하지 않으므로 current script에는 정확한 training resume 기능이 없다.

## 16. Output directory 계산

명시적인 `--output_dir`이 없으면 세 script는 같은 규칙으로 경로를 계산한다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

`network_head`는 `<network>_<head>`이고 experiment 이름은 다음 규칙이다.

```text
<model>_bs<batch_size>_ep<max_epochs>_<network>_<head>
```

evaluation과 prediction도 현재 command의 `batch_size`와 `max_epochs`로 default 경로를 계산한다. checkpoint
파일 자체는 다른 경로를 지정할 수 있지만 결과 저장 위치가 의도와 달라질 수 있으므로, 학습과 같은
identity option을 유지하거나 `--output_dir`을 명시하는 것이 좋다.

## 17. 재현성의 범위

seed는 dataset split, subset sampling, train dataloader shuffle generator에 사용된다. 같은 CSV 목록과 순서,
같은 seed를 유지하면 sample partition을 동일하게 구성하는 데 도움이 된다.

그러나 random augmentation은 Python `random`을 사용하고, model initialization과 GPU 연산에 대한 전역 seed
설정은 current scripts에 없다. 따라서 `--seed` 하나만으로 training 결과 전체의 bit-level 재현성이
보장되지는 않는다. 현재 seed는 주로 data selection 재현성으로 이해해야 한다.

## 18. Batch runner

`scripts/batch_run.py`는 `scripts/batch_config.py`의 `CONFIGS`를 순회하며 각 작업을 subprocess로 실행한다.
mode는 `train`, `evaluate`, `predict` 중 하나다.

train mode는 자동으로 `--save`를 붙인다. evaluate와 predict mode의 config에 checkpoint가 없으면 config로
계산한 output directory 아래 `model.pth`를 사용한다. 한 config가 실패해도 다음 config를 계속 실행하고,
마지막에 하나라도 실패했으면 exit code 1을 반환한다.

현재 batch config에는 CLI의 모든 공통 option이 전달되지 않는다. `PASS_KEYS`에 정의된 option만 subprocess
command에 포함되므로 dataset, csv path, seed처럼 목록에 없는 값은 script default를 사용한다.

## 19. 실패 진단

실행 단계별 대표 증상과 점검 위치는 다음과 같다.

| 증상 | 먼저 확인할 내용 |
| --- | --- |
| CSV open 또는 key error | CSV 경로와 필수 column 이름 |
| image file not found | `image_dir`과 `image_name` 결합 결과 |
| dataloader worker error | `--num_workers 0`으로 원인 message 재확인 |
| target과 output shape mismatch | image size, model, head, preprocessor resolution |
| unsupported network 또는 capability error | network와 head 조합, feature stages 제공 여부 |
| early stopping disabled | validation result에 `iou` key가 있는지 |
| checkpoint key mismatch | 학습과 평가의 model, network, head 일치 여부 |
| checkpoint가 생성되지 않음 | training command에 `--save`가 있는지 |
| metrics는 좋지만 실패 sample 존재 | `sr`과 `predictions.csv`를 함께 확인 |
| 같은 seed인데 결과가 달라짐 | augmentation, initialization, GPU determinism 범위 |

## 20. Code mapping

전체 lifecycle의 source 위치는 다음과 같다.

| 단계 | source |
| --- | --- |
| CLI default와 output path | `scripts/config.py` |
| training 진입점 | `scripts/train.py` |
| evaluation 진입점 | `scripts/evaluate.py` |
| prediction 진입점 | `scripts/predict.py` |
| split과 object factory | `src/core/factory.py` |
| dataset과 subset | `src/data/dataset.py` |
| batch 정책 | `src/data/dataloader.py` |
| wrapper step과 warmup hook | `src/models/base/wrapper.py` |
| epoch와 early stopping | `src/core/trainer.py` |
| test metric | `src/core/evaluator.py` |
| sample CSV | `src/core/predictor.py` |
| checkpoint IO | `src/utils/io.py` |

## 21. 핵심 요약

training은 model-specific loss로 parameter를 갱신하고 validation `iou`로 best state를 선택한다. 학습이
끝나면 best state를 복원하며, `--save`가 있을 때만 checkpoint로 기록한다. evaluation은 checkpoint를
같은 assembly로 다시 만들고 공통 6개 metric을 계산한다. prediction은 같은 final corner를 sample 단위로
저장한다. 세 작업을 비교 가능한 하나의 실험으로 묶으려면 CSV 순서, seed, model, network, head와 output
identity를 일관되게 유지해야 한다.
