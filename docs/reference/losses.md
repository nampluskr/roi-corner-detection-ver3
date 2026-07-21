# Loss Reference

loss는 model raw output과 preprocessor가 만든 target 사이에서 계산된다. 같은 final corner contract를
공유해도 raw output 표현이 다르면 loss를 그대로 공유할 수 없다.

## Shared Losses

현재 `src.components.losses`에 구현된 loss는 다음과 같다.

| loss | 입력 표현 | 역할 |
| --- | --- | --- |
| `BCELoss` | binary mask logits | pixel-wise binary classification |
| `DiceLoss` | binary mask logits | foreground overlap 보완 |
| `BCEDiceLoss` | binary mask logits | BCE와 Dice의 결합 |
| `DeepSupervisedSmoothL1Loss` | refinement corner sequence | 모든 refinement step supervision |
| `FocalLoss` | detection class map | sparse positive cell classification |
| `HeatmapMSELoss` | Gaussian map logits | sigmoid map regression |
| `HeatmapFocalLoss` | sparse dense map logits | peak와 ridge background 억제 |
| `SmoothL1Loss` | detection regression map | positive cell box or point regression |
| `WingLoss` | coordinate logits | 작은 coordinate error에 민감한 regression |

`BaseLoss`는 batch mean tensor를 반환하면서 running mean도 누적한다. wrapper는 각 loss의 `weight`를
곱해 total training loss를 만들고, trainer는 이름별 running mean을 history에 기록한다.

## Model Defaults

기본 wrapper loss 조합은 다음과 같다.

| model | default loss |
| --- | --- |
| `reg` | `WingLoss` with sigmoid output |
| `seg`, `torchseg` | `BCELoss` and `DiceLoss` |
| `det` | `FocalLoss` and masked `SmoothL1Loss` |
| `peak`, `ridge` | `HeatmapFocalLoss` |
| `gcn` | `DeepSupervisedSmoothL1Loss` |
| `hybrid` | `BCEDiceLoss` |
| `torchdet`, `yolo`, `detr` | native library loss |

native whole-model loss는 project `BaseLoss`와 같은 implementation이 아니다. wrapper가 native loss를
backpropagate하고, 로그를 위해 detached component loss를 running result로 수집한다.

## Selection Notes

mask logits에는 sigmoid를 loss 내부에서 처리하는 BCE 계열을 사용한다. coordinate regression은
postprocessor가 아닌 loss에서 sigmoid를 적용해 normalized target과 비교한다. detection regression은
positive grid cell만 loss를 받으므로 class imbalance 처리는 classification focal loss와 함께 해석한다.
