# Training Workflow Guide

이 문서는 새 dataset과 model 조합을 작은 실행으로 검증하고, 본 학습, checkpoint 평가, sample prediction,
비교 실험으로 확장하는 권장 절차를 설명한다. command 하나가 정상 종료되는 것과 신뢰할 수 있는 실험
결과를 얻는 것은 다르다. data identity, model assembly, output identity를 함께 기록해야 결과를 다시
확인할 수 있다.

## 1. Workflow의 목표

권장 workflow는 다음 질문에 순서대로 답한다.

1. data가 current contract에 맞는가?
2. 선택한 model assembly가 한 batch에서 동작하는가?
3. loss와 metric이 여러 epoch 동안 계산되는가?
4. best weight가 checkpoint로 저장되는가?
5. 같은 test split에서 공통 metric과 sample prediction을 만들 수 있는가?
6. 비교할 두 실험 사이에서 변경한 축이 명확한가?

앞 단계가 확인되지 않은 상태에서 긴 학습을 시작하면 data 또는 assembly 오류를 늦게 발견하고 계산 시간을
낭비할 수 있다.

## 2. 전체 권장 순서

한 실험의 전체 흐름은 다음과 같다.

```text
environment check
-> dataset audit
-> one-epoch smoke training
-> inspect log and history
-> saved-checkpoint evaluation
-> sample-level prediction review
-> full training
-> final evaluation and comparison
```

smoke run과 full run은 목적이 다르므로 output directory를 분리한다. smoke result가 좋아 보이더라도 적은
sample에서는 일반화 성능을 의미하지 않는다.

## 3. Step 1: 환경 확인

먼저 conda environment와 project root를 맞춘다.

```bash
conda activate pytorch_env
cd <project-root>
```

CPU로 pipeline만 확인할 수 있지만 external model과 큰 backbone은 매우 느릴 수 있다. CUDA를 사용할 경우
현재 environment에서 PyTorch가 GPU를 인식하는지 확인한다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

external model은 torchvision, timm, transformers, ultralytics 같은 optional dependency와 local weight가
필요할 수 있다. 처음 pipeline을 확인할 때는 별도 pretrained file이 필요 없는 `reg/custom/gap` 조합이
가장 단순하다.

## 4. Step 2: Dataset audit

CSV를 학습에 넣기 전에 다음 계약을 확인한다.

- 필수 header는 `image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4`다.
- image path는 `image_dir`과 `image_name`을 결합해 존재해야 한다.
- corner는 `[0, 1]` normalized value다.
- 순서는 `TL`, `TR`, `BR`, `BL`이다.
- 모든 coordinate는 NaN이나 infinity가 아닌 finite value여야 한다.

몇 개의 sample을 image에 그려 보는 검사는 숫자 범위 검사보다 중요하다. 좌표 순서가 틀려도 모두 0과 1
사이에 있을 수 있기 때문이다. 자세한 준비 방법은 [Dataset Format Guide](01-dataset-format.md)를 따른다.

## 5. Step 3: 실험 identity 결정

학습 전에 최소한 다음 값을 기록한다.

| 범주 | 기록할 값 | 이유 |
| --- | --- | --- |
| data | CSV 목록과 순서, dataset label | 어떤 sample pool을 사용했는지 확인 |
| split | seed와 size limit | 같은 train, valid, test 재구성 |
| assembly | model, network, head | 같은 architecture와 표현 재구성 |
| input | image size | target와 output scale 확인 |
| lifecycle | batch size, max epochs, patience, warmup | 학습 조건 비교 |
| runtime | device와 주요 dependency version | 환경 차이 추적 |
| output | 명시적 directory | file 충돌 방지 |

automatic experiment name에는 이 값이 모두 들어가지 않는다. seed, CSV, batch size, epoch가 다르지만
dataset, model, network, head가 같으면 같은 directory가 된다. 중요한 실험은 다음처럼 사람이 구분할 수
있는 `--output_dir`을 사용한다.

```text
outputs/public/reg/smoke_seed42/
outputs/public/reg/full_seed42/
```

## 6. Step 4: One-epoch smoke training

먼저 8개 train sample과 4개 valid sample로 한 epoch를 실행한다. train sample은 batch size 이상이어야
한다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --patience 1 --warmup_epochs 0 \
  --num_workers 0 \
  --output_dir outputs/public/reg/smoke_seed42 \
  --save
```

smoke run에서 확인할 것은 높은 IoU가 아니다. 다음 조건을 본다.

1. CSV와 train, valid image가 정상적으로 열린다.
2. train progress가 0 batch가 아니라 실제로 진행된다.
3. forward, target preprocessing, loss backward가 shape 오류 없이 끝난다.
4. validation result에 monitor인 `iou`가 존재한다.
5. best weight가 복원되고 `model.pth`가 생성된다.
6. loss와 metric 숫자가 NaN이나 infinity가 아니다.

## 7. Step 5: Smoke output 확인

정상적인 smoke output directory에는 다음 file이 있다.

```text
outputs/public/reg/smoke_seed42/
├── history.json
├── model.pth
└── run.log
```

`run.log`에서 epoch result, learning rate, best weight restore message를 확인한다. `history.json`의 train과
valid list 길이가 모두 1인지 확인한다. `model.pth`가 없다면 command에 `--save`가 있었는지 먼저 확인한다.

smoke run의 loss가 한 번 계산됐다는 사실만으로 target 의미가 맞다고 단정할 수 없다. 다음 prediction
단계에서 좌표 순서와 값 범위를 확인해야 한다.

## 8. Step 6: Smoke checkpoint 평가

학습에 사용한 CSV, seed, assembly와 image size를 유지하고 test size를 작게 지정한다.

```bash
python scripts/evaluate.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/smoke_seed42/model.pth \
  --output_dir outputs/public/reg/smoke_seed42
```

`metrics.json`에 `iou`, `mcd`, `maxcd`, `pck_002`, `pck_005`, `sr`가 모두 있는지 확인한다. smoke model은
성능이 낮아도 괜찮지만 값이 계산 가능해야 한다.

## 9. Step 7: Sample prediction 확인

같은 checkpoint로 prediction CSV를 만든다.

```bash
python scripts/predict.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/smoke_seed42/model.pth \
  --output_dir outputs/public/reg/smoke_seed42
```

`predictions.csv`에서 다음을 확인한다.

- target과 prediction이 모두 8개 coordinate를 가진다.
- prediction이 모두 같은 point로 붕괴하지 않았는지 확인한다.
- `pred_x1,y1`부터 `pred_x4,y4`까지 순서가 target convention과 같은지 확인한다.
- `success=False` row가 있는지 확인한다.
- `success=True`라도 zero 또는 center fallback이 반복되는지 확인한다.

가능하면 몇 row를 image에 다시 그려 polygon 방향과 ROI 위치를 직접 확인한다. current prediction CSV에는
image path가 없으므로 test split index를 원본 path와 자동 연결하는 별도 도구는 제공되지 않는다.

## 10. Step 8: Model 선택

smoke pipeline이 확인된 뒤 문제 표현을 선택한다. 대표적인 선택 기준은 다음과 같다.

| 목표 또는 가정 | 먼저 볼 model | 이유 |
| --- | --- | --- |
| 가장 단순한 coordinate baseline | `reg` | corner를 직접 회귀 |
| ROI 내부 area supervision | `seg` | polygon mask를 학습 |
| corner별 dense localization | `peak` | 네 Gaussian peak map |
| edge line supervision | `ridge` | 네 ridge map의 교차점 복원 |
| detector 방식 비교 | `det` | corner class와 local regression |
| iterative geometry refinement | `gcn` | graph sequence로 corner 보정 |
| mask와 classical geometry 결합 | `hybrid` | learned mask를 line geometry로 decode |
| library whole-model baseline | external models | native architecture와 loss 사용 |

모든 model이 동일한 loss scale을 사용하지 않는다. model 선택은 train loss 숫자가 작아 보이는지를 기준으로
하지 않고 standalone common metric과 prediction 품질로 비교한다. 자세한 원리는
[Model Guide](../models/README.md)를 참고한다.

## 11. Step 9: Compatible assembly 확인

model마다 필요한 head와 feature capability가 다르다. 예를 들어 `seg`는 `mask` head와 multi-stage feature를
요구한다. `reg` global head에 적합한 transformer가 dense decoder에도 반드시 적합한 것은 아니다.

새 assembly는 먼저 smoke size로 실행한다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --model seg --network resnet18 --head mask \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --num_workers 0 \
  --output_dir outputs/public/seg/smoke_seed42 \
  --save
```

pretrained network는 local weight file이 필요할 수 있다. constructor 단계에서 weight error가 나면 data나
trainer 문제로 보기 전에 model 문서의 weight 경로와 dependency를 확인한다.

## 12. Step 10: Full training으로 확장

smoke 검증 후 sample 수, batch size, epoch를 목적에 맞게 늘린다. 한 번에 여러 축을 바꾸지 않으면 문제가
생겼을 때 원인을 찾기 쉽다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model seg --network resnet18 --head mask \
  --image_size 224 --batch_size 4 \
  --train_size 2000 --valid_size 1000 \
  --max_epochs 50 --patience 8 --warmup_epochs 1 \
  --num_workers 4 \
  --output_dir outputs/public/seg/full_seed42 \
  --save
```

current CLI에서 split 전체를 의미하는 특별한 `all` 또는 `None` 문자열은 없다. split보다 충분히 큰 integer를
주면 사용 가능한 sample 전체가 선택된다.

## 13. Warmup 사용

pretrained backbone을 바로 큰 learning rate로 갱신하면 기존 feature를 급격히 잃을 수 있다. warmup phase는
초기 epoch 동안 backbone을 freeze하고 task layer를 먼저 학습한 뒤 전체 network를 연다.

다음 command는 wrapper에 한 epoch warmup을 요청한다.

```bash
python scripts/train.py ... --warmup_epochs 1
```

모든 wrapper가 요청값을 실제 phase freeze로 적용하는 것은 아니다. current `torchseg`처럼 option은 받지만
`applied_warmup_epochs`가 0인 구현도 있다. model wrapper source와 log의 learning rate 변화를 함께 확인한다.

warmup 전환 시 optimizer와 scheduler를 새로 만들므로 optimizer internal state는 phase 사이에 이어지지
않는다.

## 14. Early stopping 해석

trainer는 validation `iou`가 이전 best보다 `1e-4` 이상 좋아지는지 확인한다. 개선되지 않은 epoch가
`patience`만큼 연속되면 중단한다.

예를 들어 patience가 3이면 best 이후 세 epoch가 모두 개선되지 않을 때 세 번째 epoch 뒤에 멈춘다. 종료
후 model은 best epoch weight로 복원된다. history에는 best 이후의 non-improving epoch도 포함된다.

validation result에 `iou` key가 없으면 early stopping을 비활성화하고 max epoch까지 계속한다. terminal과
`run.log`에서 관련 message를 확인한다.

## 15. Full checkpoint 평가

본 학습이 끝나면 full test size로 evaluation과 prediction을 다시 실행한다. 학습 command의 data와 assembly
identity를 유지한다.

```bash
python scripts/evaluate.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model seg --network resnet18 --head mask \
  --image_size 224 --batch_size 4 --test_size 1000 \
  --checkpoint outputs/public/seg/full_seed42/model.pth \
  --output_dir outputs/public/seg/full_seed42

python scripts/predict.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model seg --network resnet18 --head mask \
  --image_size 224 --batch_size 4 --test_size 1000 \
  --checkpoint outputs/public/seg/full_seed42/model.pth \
  --output_dir outputs/public/seg/full_seed42
```

metric file만 만들고 끝내지 말고 prediction CSV에서 worst sample과 fallback pattern을 확인한다.

## 16. 비교 실험 설계

두 실험을 공정하게 비교하려면 한 번에 한 축을 바꾸는 것이 좋다. 비교 가능한 예시는 다음과 같다.

| 비교 목적 | 고정할 값 | 바꿀 값 |
| --- | --- | --- |
| model 표현 비교 | CSV, split, network capability, training scale | model과 compatible head |
| backbone 비교 | CSV, split, model, head, epoch | network |
| warmup 효과 | CSV, split, assembly, epoch | warmup epochs |
| data 양 효과 | CSV, split seed, assembly | train size |
| detector target 비교 | CSV, split, model과 network | box 또는 point head |

model이 다르면 optimizer와 loss scale도 다를 수 있으므로 같은 epoch 수가 반드시 같은 학습량을 뜻하지는
않는다. 최종 비교는 같은 test sample에서 `iou`, distance, PCK, success rate와 runtime을 함께 본다.

## 17. Metric 읽는 순서

한 숫자만 선택하지 말고 다음 순서로 해석한다.

1. `sr`로 non-finite failure 비율을 확인한다.
2. `iou`로 ROI polygon 전체 overlap을 확인한다.
3. `mcd`로 평균 corner localization을 확인한다.
4. `maxcd`로 sample당 가장 나쁜 corner의 경향을 확인한다.
5. `pck_002`, `pck_005`로 정확도 threshold별 비율을 확인한다.
6. `predictions.csv`에서 finite fallback과 corner ordering 오류를 확인한다.

NaN prediction은 일반 distance와 IoU 평균에서 제외될 수 있으므로 `sr`가 낮은 model의 나머지 metric만
직접 비교하면 결과를 과대평가할 수 있다.

## 18. Batch run을 사용할 때

검증된 config를 여러 개 순차 실행할 때 `batch_config.py`와 `batch_run.py`를 사용할 수 있다.

```bash
python scripts/batch_run.py --mode train
python scripts/batch_run.py --mode evaluate
python scripts/batch_run.py --mode predict
python scripts/batch_run.py --mode all
```

batch run 전에 active `CONFIGS`만 남았는지 확인한다. current runner는 config dict의 `dataset`, `csv_path`,
`network`, `head`, lifecycle option을 하위 script로 전달하므로 stage별로 다른 CSV와 dataset을 config에서
지정할 수 있다. `seed` 같은 일부 option은 아직 전달하지 않으므로 필요하면 command를 직접 실행하거나
runner를 먼저 확장한다.

한 config가 실패해도 runner는 다음 config를 계속 실행하고 마지막 summary에서 실패를 보고한다. terminal
중간의 `[FAIL]`과 최종 exit code를 모두 확인한다.

`all` mode는 config별 train, evaluate, predict 순서가 아니라 전체 train, 전체 evaluate, 전체
predict 순서로 실행한다. 별도 설정 파일을 사용하려면 top-level `CONFIGS`를 정의하고 다음과
같이 지정한다.

```bash
python scripts/batch_run.py --mode all --config configs/public.py
```

`--config`를 생략하면 `scripts/batch_config.py`를 사용한다. 상대 path는 current working
directory와 project root 순서로 탐색한다. dataset stage별 config 파일은 `configs/` 아래에 있으며,
stage별 학습과 평가, 추론만 수행하는 CLI 시나리오는 [Use Cases](06-use-cases.md)에서 다룬다.

여러 실험의 `metrics.json`을 한 표로 모을 때는 `scripts/collect_metrics.py`를 사용한다. 이 script는
`outputs/` 아래 experiment directory를 순회하며 metric을 pandas DataFrame으로 모아 CSV로 저장한다.

```bash
python scripts/collect_metrics.py --dataset public --output outputs/public/metrics_summary.csv
```

## 19. 재현성 체크리스트

현재 seed가 data selection에 주로 적용된다는 제한을 고려해 다음 정보를 함께 보관한다.

- CSV file 목록, row order와 가능한 경우 file hash
- seed와 train, valid, test size
- model, network, head, image size
- batch size, epoch, patience, warmup
- checkpoint와 history, metrics, predictions
- source commit identifier
- Python, PyTorch와 optional library version
- 사용 device와 주요 runtime condition

current script는 이 metadata를 자동 manifest로 저장하지 않는다. output directory 이름이나 별도 실험 기록으로
관리해야 한다.

## 20. 흔한 문제의 진단 순서

문제가 생기면 다음과 같이 범위를 좁힌다.

| 단계 | 질문 | 조치 |
| --- | --- | --- |
| data | 한 sample을 열고 corner를 읽을 수 있는가 | worker 0, 작은 subset |
| assembly | constructor가 network와 head를 허용하는가 | model 문서와 factory 확인 |
| forward | image와 raw output shape가 맞는가 | image size 224 유지 |
| target | preprocessor target shape가 raw output과 맞는가 | model별 문서 확인 |
| optimization | loss가 finite이고 backward가 되는가 | 작은 batch와 custom network |
| validation | `iou`가 result에 있는가 | early stopping log 확인 |
| checkpoint | best state가 저장됐는가 | `--save`, path, file 확인 |
| evaluation | assembly와 split이 같은가 | command option 대조 |
| prediction | finite지만 잘못된 fallback인가 | sample row와 시각화 확인 |

## 21. 현재 제약

workflow를 운영할 때 알아야 할 current limitation은 다음과 같다.

- training resume를 위한 optimizer, scheduler, epoch checkpoint는 없다.
- checkpoint에는 model state만 있고 config metadata가 없다.
- seed는 전체 training의 bit-level determinism을 보장하지 않는다.
- non-default image size는 wrapper target와 scale에 반영되지 않을 수 있다.
- automatic experiment name은 seed, CSV, size limit 등 모든 실험 변수를 포함하지 않는다.
- predictor는 image path와 상세 postprocess failure reason을 저장하지 않는다.
- batch runner는 `dataset`과 `csv_path`를 포함해 대부분의 option을 전달하지만 `seed` 등 일부 option은
  아직 config에서 전달하지 않는다.

## 22. Code mapping

workflow 각 단계의 source는 다음과 같다.

| 단계 | source |
| --- | --- |
| common option과 identity | `scripts/config.py` |
| smoke와 full training | `scripts/train.py` |
| epoch, scheduler, early stopping | `src/core/trainer.py` |
| wrapper lifecycle | `src/models/base/wrapper.py` |
| common evaluation | `scripts/evaluate.py`, `src/core/evaluator.py` |
| sample prediction | `scripts/predict.py`, `src/core/predictor.py` |
| batch experiment | `scripts/batch_config.py`, `configs/`, `scripts/batch_run.py` |
| metric 집계 | `scripts/collect_metrics.py` |

## 23. 핵심 요약

신뢰할 수 있는 실험은 data audit, one-epoch smoke run, checkpoint evaluation, sample prediction 확인을 통과한
뒤 규모를 늘린다. 비교 실험에서는 CSV, seed, split size와 assembly identity를 고정하고 한 축씩 바꾼다.
best checkpoint, history, common metric, prediction CSV를 한 묶음으로 보관하며, metric 숫자와 sample failure를
함께 해석해야 한다.
