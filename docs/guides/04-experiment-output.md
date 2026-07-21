# Experiment Output Guide

이 문서는 training, evaluation, prediction이 만드는 directory와 file의 의미를 설명한다. output은 단순한
결과 저장 장소가 아니라 어떤 model assembly와 실행 조건에서 나온 결과인지 추적하는 실험 기록이다.
current implementation이 자동으로 기록하지 않는 metadata도 있으므로 경로와 file을 함께 관리해야 한다.

## 1. 자동 output directory

`--output_dir`을 지정하지 않으면 다음 규칙으로 directory를 만든다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

각 segment의 의미는 다음과 같다.

| segment | 생성 규칙 | 예시 |
| --- | --- | --- |
| `dataset` | `--dataset` 문자열 | `public` |
| `model` | `--model` 문자열 | `seg` |
| `network_head` | `<network>_<head>` | `resnet18_mask` |
| `exp_name` | `<model>_bs<batch_size>_ep<max_epochs>_<network>_<head>` | `seg_bs4_ep50_resnet18_mask` |

예를 들어 public dataset label, segmentation, ResNet-18, mask head, batch size 4, max epoch 50은 다음 path를
사용한다.

```text
outputs/public/seg/resnet18_mask/seg_bs4_ep50_resnet18_mask/
```

## 2. `dataset` segment의 의미

`--dataset`은 data loader가 CSV를 자동 선택하는 기능이 아니다. output을 `public`, `synthetic`, `measured`
같은 논리 stage로 구분하기 위한 문자열이다. 실제 sample은 `--csv_path`가 결정한다.

따라서 서로 다른 CSV를 사용하면서 같은 dataset label과 나머지 option을 쓰면 같은 output directory가 될
수 있다. 중요한 실험은 CSV identity를 별도로 기록하거나 명시적 output path를 사용한다.

## 3. 자동 이름에 포함되지 않는 값

automatic `exp_name`에는 다음 값이 포함되지 않는다.

- CSV path와 file version
- seed
- image size
- train, valid, test size
- patience와 warmup epochs
- optimizer와 learning rate
- device와 dependency version

이 값만 달리한 두 실행은 같은 directory의 `history.json`, `model.pth`, `metrics.json`, `predictions.csv`를
덮어쓸 수 있다. `run.log`는 append mode로 열리므로 이전 실행 log와 새 실행 log가 같은 file에 이어질 수
있다.

## 4. 명시적 output directory

실험을 명확히 구분하려면 다음처럼 `--output_dir`을 지정한다.

```bash
python scripts/train.py \
  --csv_path /data/gt_corners.csv \
  --model reg --network resnet18 --head spatial \
  --seed 42 --max_epochs 50 --save \
  --output_dir outputs/public/reg/resnet18_spatial/full_seed42
```

명시 경로는 automatic rule 전체를 대체한다. directory가 없으면 logger와 save 함수가 생성한다. checkpoint를
별도 path로 저장할 때는 parent directory가 있는 path를 사용하는 것이 좋다.

## 5. 단계별 생성 file

한 실험 directory에 생성될 수 있는 file은 다음과 같다.

| file | 생성 작업 | 항상 생성되는가 | 내용 |
| --- | --- | --- | --- |
| `run.log` | training | trainer가 생성되면 기록 | timestamp log |
| `history.json` | training | training 정상 종료 시 생성 | epoch별 train과 valid result |
| `model.pth` | training | `--save`가 있을 때만 생성 | best model state dictionary |
| `metrics.json` | evaluation | evaluation 정상 종료 시 생성 | test split 공통 scalar metric |
| `predictions.csv` | prediction | prediction 정상 종료 시 생성 | sample별 target과 final corner |

evaluation의 `Evaluator`와 prediction의 `Predictor`는 file logger를 만들지 않는다. 따라서 current
`evaluate.py`와 `predict.py` 실행만으로 `run.log`에 결과가 추가되지는 않는다. file 존재 여부만으로 특정
단계를 실행했다고 판단하지 않는다.

## 6. `run.log`

trainer logger는 terminal에는 간단한 message를, file에는 timestamp와 level을 기록한다. 대표 내용은 다음과
같다.

```text
2026-07-22 10:00:00 [INFO] [ 1/5] loss=... iou=... | loss=... iou=... | lr=...
2026-07-22 10:03:00 [INFO] restored best weights from epoch 3 (iou=...)
2026-07-22 10:03:01 [INFO] model saved to outputs/.../model.pth
```

한 directory에서 training을 다시 실행하면 file handler의 append 동작으로 이전 log 뒤에 새 기록이 붙는다.
어느 실행에 해당하는지 구분하기 어려울 수 있으므로 반복 실험은 directory를 나누는 것이 좋다.

log에 표시되는 learning rate는 optimizer 마지막 parameter group의 값이다. 여러 parameter group을 가진
optimizer의 모든 learning rate를 보여 주는 것은 아니다.

## 7. `history.json` 구조

history는 train과 valid 아래에 metric 이름별 epoch list를 둔다. model마다 loss key와 wrapper metric key가
다르므로 정확한 이름은 달라질 수 있다.

개념적인 형식은 다음과 같다.

```json
{
  "train": {
    "loss_name": [0.9, 0.7, 0.5],
    "iou": [0.2, 0.4, 0.6]
  },
  "valid": {
    "loss_name": [0.8, 0.65, 0.6],
    "iou": [0.25, 0.45, 0.55]
  }
}
```

list index 0은 epoch 1을 의미한다. early stopping이 발생하면 list 길이는 `max_epochs`보다 짧다. best epoch
이후 개선되지 않은 epoch도 중단 전까지 기록된다.

history에는 best epoch 번호, seed, CLI config, wall-clock time가 별도 field로 저장되지 않는다. best epoch는
`run.log`의 restore message와 valid monitor list를 함께 확인한다.

## 8. Training loss 해석

history의 loss는 model-specific training objective다. `reg` coordinate loss와 `seg` BCE 또는 Dice loss,
external detector native loss는 scale과 의미가 다르다. 따라서 서로 다른 model의 loss 숫자를 직접 비교해
어느 model이 더 정확하다고 판단하면 안 된다.

같은 model과 config 안에서는 다음 경향을 확인할 수 있다.

- train loss가 전혀 변하지 않으면 optimizer, target, gradient를 점검한다.
- train loss는 줄지만 valid metric이 나빠지면 overfitting 가능성을 본다.
- train과 valid가 모두 NaN이면 input, target, loss numerical stability를 점검한다.
- loss는 줄지만 final corner metric이 개선되지 않으면 postprocessor와 target 표현을 점검한다.

## 9. `model.pth`

checkpoint는 다음 값만 저장한다.

```python
wrapper.model.state_dict()
```

포함되는 것과 포함되지 않는 것은 다음과 같다.

| 포함 | 포함되지 않음 |
| --- | --- |
| named parameter tensor | model, network, head 문자열 |
| registered buffer tensor | optimizer state |
| model module state | scheduler state |
|  | current epoch와 history |
|  | seed, CSV, image size |

training 종료 시 early stopping이 선택한 best state를 먼저 model에 복원한다. 이후 `--save`를 사용하면
`model.pth`는 best validation epoch weight를 담는다.

평가 시 same model, network, head로 object를 다시 만든 뒤 checkpoint를 load한다. assembly가 다르면 key나
shape mismatch가 발생한다. config metadata가 checkpoint에 없으므로 directory와 별도 기록을 함께 보관해야
한다.

## 10. `metrics.json`

standalone evaluator는 final corner prediction에 대해 여섯 metric을 계산한다. 예시 구조는 다음과 같다.

```json
{
  "iou": 0.75,
  "mcd": 0.03,
  "maxcd": 0.05,
  "pck_002": 0.45,
  "pck_005": 0.82,
  "sr": 0.98
}
```

각 값은 다음 의미를 가진다.

| key | 범위 또는 단위 | 해석 |
| --- | --- | --- |
| `iou` | 보통 0부터 1 | polygon overlap, 높을수록 좋음 |
| `mcd` | normalized Euclidean distance | 대응 corner 평균 오차, 낮을수록 좋음 |
| `maxcd` | normalized Euclidean distance | sample마다 가장 큰 corner 오차의 평균, 낮을수록 좋음 |
| `pck_002` | 0부터 1 | 0.02 이내 corner 비율, 높을수록 좋음 |
| `pck_005` | 0부터 1 | 0.05 이내 corner 비율, 높을수록 좋음 |
| `sr` | 0부터 1 | finite prediction sample 비율, 높을수록 좋음 |

각 metric의 정확한 수식, NaN과 infinity aggregation은 [Metric Reference](../reference/02-metrics.md)를
참고한다.

## 11. Polygon IoU

predicted quadrilateral을 $P$, target quadrilateral을 $T$라고 하면 IoU는 다음과 같다.

$$
IoU = \frac{|P \cap T|}{|P \cup T|}
$$

완전히 겹치면 1이고 겹치지 않으면 0이다. corner 하나의 작은 이동도 polygon shape와 area에 따라 IoU에
다르게 영향을 준다. 큰 ROI와 작은 ROI에서 같은 normalized point 오차가 동일한 IoU 감소를 만들지는 않는다.

current implementation은 ordered quadrilateral을 대상으로 polygon clipping을 수행한다. corner 순서가
틀리거나 self-intersection이 있으면 기대하지 않은 area가 계산될 수 있다.

## 12. Corner distance

corner $i$의 prediction을 $p_i=(x_i,y_i)$, target을 $t_i=(x_i^*,y_i^*)$라고 하면 distance는 다음과 같다.

$$
d_i = \sqrt{(x_i-x_i^*)^2 + (y_i-y_i^*)^2}
$$

MCD는 한 sample의 네 distance 평균을 다시 dataset 평균으로 집계한다. MaxCD는 각 sample에서 가장 큰
distance를 선택한 뒤 dataset 평균을 낸다. 따라서 MaxCD가 크면 네 점 중 하나가 반복해서 크게 빗나가는
문제를 찾는 데 도움이 된다.

distance는 normalized 단위다. 224 square image에서 0.05는 한 축으로만 차이가 난 경우 약 11.2 pixel에
해당하지만, 두 축 차이가 함께 있으면 Euclidean 관계를 사용해야 한다.

## 13. PCK

PCK는 distance가 threshold 이하인 corner 비율이다.

$$
PCK(\tau) = \frac{1}{4N}\sum_{n=1}^{N}\sum_{i=1}^{4} \mathbb{1}[d_{n,i} \le \tau]
$$

`pck_002`는 엄격한 localization, `pck_005`는 더 완화된 localization을 본다. 두 값의 차이가 크면 대다수
corner가 대략 맞지만 매우 정확한 threshold에는 들지 못한다는 의미일 수 있다.

## 14. Success rate와 invalid sample

`sr`은 한 sample의 여덟 coordinate가 모두 finite이면 성공으로 센다.

```text
finite values only -> success
contains NaN or infinity -> failure
```

zero corners, image center fallback, 범위 밖 finite coordinate도 success일 수 있다. success는 정확도 metric이
아니라 계산 가능한 output을 만들었는지 보는 최소 validity metric이다.

또한 base metric은 NaN prediction sample을 IoU와 distance 평균에서 제외한다. `sr`가 낮은데 나머지 metric이
좋아 보이면 실패 sample이 평균에서 빠진 영향을 의심해야 한다.

## 15. `predictions.csv` schema

prediction CSV의 column은 세 부분으로 나뉜다.

```text
index,success,failure_reason,
target_x1,target_y1,...,target_x4,target_y4,
pred_x1,pred_y1,...,pred_x4,pred_y4
```

각 부분의 의미는 다음과 같다.

| column group | 의미 |
| --- | --- |
| `index` | test dataloader에서 순차적으로 부여한 0-based row 번호 |
| `success` | prediction 전체가 finite인지 여부 |
| `failure_reason` | 실패면 `invalid_prediction`, 성공이면 빈 문자열 |
| `target_*` | normalized ordered target corner 8개 값 |
| `pred_*` | normalized ordered final prediction 8개 값 |

`index`는 원본 CSV row index나 image file 이름이 아니다. current predictor는 image path, raw output,
confidence, postprocessor의 상세 failure reason을 기록하지 않는다.

## 16. Prediction row 읽기

한 row를 볼 때 다음 순서로 확인한다.

1. `success`와 `failure_reason`을 확인한다.
2. target과 prediction이 모두 `TL`, `TR`, `BR`, `BL` 순서인지 확인한다.
3. prediction 값이 대략 `[0, 1]` 범위에 있는지 확인한다.
4. 각 대응 corner 차이를 확인한다.
5. 네 점이 교차하지 않는 polygon을 만드는지 확인한다.
6. 여러 row가 모두 zero 또는 `(0.5, 0.5)` 부근인지 확인한다.

finite 여부만으로는 geometry validity를 알 수 없다. model에 따라 후보가 없을 때 center fallback이나 zero를
반환할 수 있기 때문이다.

## 17. File overwrite 동작

같은 output directory에서 작업을 다시 실행하면 file별 동작은 다음과 같다.

| file | 재실행 동작 |
| --- | --- |
| `run.log` | 기존 내용 뒤에 append될 수 있음 |
| `history.json` | 새 history로 overwrite |
| `model.pth` | 새 state dictionary로 overwrite |
| `metrics.json` | 새 metric으로 overwrite |
| `predictions.csv` | 새 row로 overwrite |

자동 backup이나 run ID는 없다. 비교해야 하는 기존 결과는 별도 directory로 구분한다.

## 18. 권장 output 보관 단위

하나의 완료된 실험은 다음 file을 함께 보관하는 것이 좋다.

```text
experiment-directory/
├── history.json
├── metrics.json
├── model.pth
├── predictions.csv
└── run.log
```

여기에 current source commit, command, CSV identity와 dependency version을 별도 기록하면 재현성이 높아진다.
current code는 command나 config manifest를 자동으로 저장하지 않는다.

## 19. 결과 비교 checklist

두 output directory를 비교하기 전에 다음 항목을 확인한다.

- 같은 CSV 목록과 row order를 사용했는가?
- 같은 seed와 test size를 사용했는가?
- 같은 normalized corner convention을 사용했는가?
- checkpoint가 각 directory의 assembly와 일치하는가?
- `sr` 차이가 일반 metric 평균에 영향을 주지 않았는가?
- prediction에 반복 fallback이나 ordering 오류가 없는가?
- automatic 이름에 포함되지 않은 option이 달라지지 않았는가?

이 조건이 다르면 metric 차이가 model 자체가 아니라 data 또는 실행 조건 차이에서 생길 수 있다.

## 20. 흔한 결과 해석 오류

대표적인 오류는 다음과 같다.

| 잘못된 해석 | 실제로 확인할 내용 |
| --- | --- |
| loss가 더 작으므로 다른 model보다 우수하다 | model별 loss scale은 직접 비교 불가 |
| IoU 하나가 높으므로 모든 corner가 정확하다 | MCD, MaxCD, PCK 확인 |
| success가 1이므로 geometry가 유효하다 | finite fallback과 polygon ordering 확인 |
| metrics file이 있으므로 최신 checkpoint 결과다 | overwrite 시점과 command 확인 |
| 같은 output path이므로 같은 data다 | CSV와 seed는 path에 포함되지 않음 |
| history 마지막 epoch가 saved weight다 | early stopping best weight가 복원됨 |
| checkpoint만 있으면 학습을 완전히 재현한다 | config와 optimizer state가 없음 |

## 21. 흔한 file 문제

output 관련 증상과 점검 방법은 다음과 같다.

| 증상 | 가능한 원인 | 점검 방법 |
| --- | --- | --- |
| directory는 있지만 `model.pth` 없음 | `--save` 누락 | train command 확인 |
| `history.json`만 최신이고 log가 길음 | JSON overwrite, log append | timestamp와 directory 분리 |
| evaluation 결과가 다른 folder에 저장 | automatic path option 불일치 | `--output_dir` 명시 |
| checkpoint load mismatch | assembly metadata 미보관 | 학습 command와 path 이름 확인 |
| metrics가 모두 0에 가까움 | failed geometry 또는 wrong split | `sr`과 prediction row 확인 |
| `sr`는 높지만 IoU가 낮음 | finite fallback 또는 잘못된 corner | CSV sample 확인 |

## 22. 현재 제약

output system의 current limitation은 다음과 같다.

- automatic naming이 전체 configuration을 표현하지 않는다.
- run별 manifest와 unique identifier가 없다.
- checkpoint에 optimizer, scheduler, epoch와 config가 없다.
- predictor output에 image path와 confidence가 없다.
- detailed failure reason이 postprocessor에서 CSV로 전달되지 않는다.
- file overwrite 전에 confirmation이나 backup을 하지 않는다.
- metric JSON은 aggregate만 저장하고 sample별 metric은 저장하지 않는다.

## 23. Code mapping

output 생성 source는 다음과 같다.

| 기능 | source |
| --- | --- |
| directory와 experiment name | `scripts/config.py` |
| training output 호출 | `scripts/train.py` |
| history와 log | `src/core/trainer.py`, `src/core/factory.py` |
| checkpoint save와 load | `src/utils/io.py` |
| aggregate metric | `src/core/evaluator.py` |
| prediction CSV | `src/core/predictor.py` |

## 24. 핵심 요약

automatic output path는 dataset label, model, network, head, batch size, max epoch만 표현한다. history는 epoch
과정, checkpoint는 best model state, metrics는 test aggregate, predictions는 sample별 final corner를
담는다. loss와 common metric을 구분하고 `sr`와 fallback을 함께 확인해야 한다. 같은 path의 재실행은 log를
이어 붙이거나 다른 file을 덮어쓸 수 있으므로 실험 directory와 metadata를 명시적으로 관리해야 한다.
