# Model Contract

모든 model은 내부 표현과 무관하게 네 개의 normalized corner를 공통 결과로 사용한다. 이 계약은
dataset, transform, wrapper, evaluator, predictor가 서로 다른 model을 같은 실행 경로에서 다룰 수 있게
한다.

## Corner Contract

corner target과 최종 prediction은 `(B, 4, 2)` float tensor 또는 array다. 좌표는 `[0, 1]` normalized
image coordinate이고 corner 순서는 `TL`, `TR`, `BR`, `BL`이다. `(x, y)`는 각각 image width와 height를
기준으로 정규화한다.

이 순서는 label CSV, geometry transform, preprocessor, postprocessor, metric에서 유지되어야 한다. flip
transform은 좌표 위치뿐 아니라 corner 순서도 다시 정렬한다.

## Model Boundary

model package는 다음 네 역할을 분리한다.

| 구성요소 | 입력 | 출력 | 책임 |
| --- | --- | --- | --- |
| `Model` | image tensor | method raw output | neural network forward |
| `Preprocessor` | normalized corners | method target | label rasterization 또는 detection label 생성 |
| `Postprocessor` | method raw output | normalized corners | decode와 geometry 복원 |
| `Wrapper` | batch | loss, metric, prediction | device, optimizer, step lifecycle |

학습 단계에서 `BaseWrapper.train_step`은 image와 target을 device로 옮기고 raw output을 계산한다.
`compute_losses`는 standard corner target을 method target으로 변환한 뒤 model별 loss를 계산한다. 평가와
예측 단계에서는 postprocessor가 raw output을 standard corner로 되돌린다.

## Feature Contract

composable model은 `FeatureExtractor`가 반환하는 `FeatureBundle`을 사용한다.

| field | 의미 | 주요 consumer |
| --- | --- | --- |
| `global_feature` | image 전체를 대표하는 vector feature | `reg`의 `gap` head |
| `spatial_feature` | 마지막 spatial feature map | `reg`의 `spatial` head |
| `stages` | encoder stage feature list | decoder, neck, dense model |

adapter는 native backbone 출력의 차이를 이 계약으로 변환한다. decoder와 neck은 `FeatureSpec`의 channel과
capability를 검증해 필요한 feature가 없는 조합을 조기에 거부한다.

## Failure Representation

현재 기본 postprocessor는 네 corner를 tensor로 반환한다. detection 계열은 특정 class 후보가 없을 때
해당 corner를 image 중심 `(0.5, 0.5)`으로 남긴다. predictor는 prediction 전체가 finite인지로 `success`를
기록하고, `failure_reason` column을 함께 보존한다.

이 규칙은 실패를 별도 exception으로 숨기지 않는다. 결과를 해석할 때는 `metrics.json`의 scalar와
`predictions.csv`의 표본별 prediction을 함께 확인한다.
