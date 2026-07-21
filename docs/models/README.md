# Model Guide

이 폴더는 ROI의 네 corner를 찾는 여러 접근법을 초보자 관점에서 설명한다. 모든 model은 같은 image와
같은 정답 corner를 사용하지만, neural network가 학습하는 중간 표현은 서로 다르다. 어떤 model은 좌표
8개를 바로 출력하고, 어떤 model은 mask나 dense map을 만든 뒤 기하학적 규칙으로 corner를 복원한다.

이 차이는 단순한 network 이름의 차이가 아니다. target을 만드는 방법, loss가 비교하는 값, raw output의
shape, postprocessor의 역할이 함께 달라진다. 이 문서를 먼저 읽으면 각 model 문서에서 무엇을 비교해야
하는지 이해할 수 있다.

## 공통 문제

한 장의 RGB image에서 quadrilateral ROI를 나타내는 네 점을 찾는 것이 공통 목표다. batch size를 `B`,
image height와 width를 `H`, `W`라고 하면 입력과 최종 출력은 다음 계약을 따른다.

| 항목 | shape | 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | ImageNet 방식으로 정규화된 RGB tensor |
| target corner | `(B, 4, 2)` | 정답 `TL`, `TR`, `BR`, `BL` 좌표 |
| final prediction | `(B, 4, 2)` | model별 postprocess가 복원한 corner |

좌표는 pixel 수가 아니라 `[0, 1]` normalized coordinate다. 예를 들어 `(0.25, 0.5)`는 image width의
25%, height의 50% 지점을 뜻한다. image size가 224라면 대략 `(56, 112)` pixel에 대응한다.

## 왜 여러 표현을 사용하는가

정답은 네 점뿐이지만, image가 제공하는 evidence는 점에만 있지 않다. ROI 내부의 texture, boundary의
연속성, 네 변이 이루는 polygon 관계도 corner 위치를 추정하는 데 도움이 된다. model은 이 evidence를
어떤 형태로 학습할지에 따라 다음 계열로 나뉜다.

| 표현 | 핵심 질문 | 해당 model |
| --- | --- | --- |
| coordinate | image 전체를 보고 좌표를 바로 예측할 수 있는가 | `reg` |
| mask | ROI 영역을 먼저 칠한 뒤 외곽에서 corner를 찾을 수 있는가 | `seg`, `hybrid`, `torchseg` |
| dense map | 각 pixel이 corner 또는 edge에 가까운 정도를 예측할 수 있는가 | `peak`, `ridge` |
| detection | corner를 class가 있는 작은 object로 취급할 수 있는가 | `det`, `torchdet`, `yolo`, `detr` |
| refinement | 초기 polygon을 만들고 네 점의 관계를 이용해 반복 보정할 수 있는가 | `gcn` |

표현이 복잡할수록 항상 성능이 좋아지는 것은 아니다. dense map과 mask는 spatial evidence를 풍부하게
사용하지만 target 생성과 postprocess가 추가된다. direct regression은 간단하지만 작은 local structure를
하나의 vector로 압축해야 한다.

## Model 문서 지도

권장 읽기 순서와 각 문서의 역할은 다음과 같다.

| 순서 | 문서 | model | 먼저 이해할 개념 |
| ---: | --- | --- | --- |
| 1 | [01-reg.md](01-reg.md) | `reg` | 회귀, logit, sigmoid |
| 2 | [02-seg.md](02-seg.md) | `seg` | binary mask, encoder-decoder |
| 3 | [03-dense-prediction.md](03-dense-prediction.md) | `peak`, `ridge` | Gaussian map, argmax, line fitting |
| 4 | [04-det.md](04-det.md) | `det` | grid cell, classification, offset |
| 5 | [05-gcn.md](05-gcn.md) | `gcn` | graph, vertex feature, iterative refinement |
| 6 | [06-hybrid.md](06-hybrid.md) | `hybrid` | learned mask와 classical geometry 결합 |
| 7 | [07-external-models.md](07-external-models.md) | `torchseg`, `torchdet`, `yolo`, `detr` | whole-model adapter와 native loss |

처음 읽는 경우에는 번호 순서대로 읽는 것이 좋다. 특정 model을 실행하는 것이 목적이라면 해당 문서의
실행 예시와 code mapping부터 확인한 뒤 앞쪽 개념 절로 돌아가도 된다.

## Model Package의 네 역할

대부분의 `src/models/<model>/` package에는 `model.py`, `preprocessor.py`, `postprocessor.py`,
`wrapper.py`가 있다. 각 파일의 책임을 구분하면 학습과 추론 흐름이 선명해진다.

| 역할 | 학습에서 하는 일 | 추론에서 하는 일 |
| --- | --- | --- |
| model | image에서 raw output을 계산한다 | 같은 forward 계산을 수행한다 |
| preprocessor | corner 정답을 model별 target으로 바꾼다 | 사용하지 않는다 |
| postprocessor | metric 계산을 위해 raw output을 corner로 바꾼다 | 최종 corner를 만든다 |
| wrapper | optimizer, loss, metric과 step 순서를 관리한다 | model과 postprocessor를 연결한다 |

학습에서는 다음 두 흐름이 만난다.

```text
image -> model -> raw output
corner target -> preprocessor -> model-specific target
raw output + model-specific target -> loss
```

추론에서는 정답이 없으므로 preprocessor와 loss가 빠진다.

```text
image -> model -> raw output -> postprocessor -> final corners
```

## Model 선택 질문

첫 model을 고를 때는 다음 질문을 순서대로 검토한다.

1. 가장 단순한 end-to-end baseline이 필요한가? 그러면 `reg`를 먼저 사용한다.
2. ROI 내부와 boundary가 corner보다 안정적으로 보이는가? 그러면 `seg`를 비교한다.
3. corner 주변의 국소 evidence가 중요한가? 그러면 `peak`를 사용한다.
4. 점보다 네 변의 연속성이 더 안정적인가? 그러면 `ridge`를 사용한다.
5. object detector의 assignment와 pretrained whole model을 활용하려는가? `det` 또는 external detector를
   검토한다.
6. 초기 corner는 얻을 수 있지만 polygon 관계를 이용한 보정이 필요한가? `gcn`을 검토한다.
7. mask는 안정적이지만 corner 복원을 학습에 맡기고 싶지 않은가? `hybrid`를 검토한다.

## 공정한 비교 방법

model을 비교할 때는 표현 외의 조건을 가능한 한 고정해야 한다. data split, seed, image size, epoch,
평가 metric이 달라지면 어느 차이가 model 표현에서 발생했는지 알기 어렵다.

동일한 model 안에서는 `network`나 `head` 한 축만 바꾼다. 서로 다른 model 사이에서는 공통 evaluator가
저장한 `metrics.json`을 사용하고, `predictions.csv`에서 corner order와 실패 표본을 함께 확인한다.
training loss의 숫자는 표현마다 scale과 의미가 다르므로 model 사이의 성능 지표로 직접 비교하지 않는다.

shared loss의 수식과 current edge case는 [Loss Reference](../reference/01-losses.md), final corner metric과
invalid sample aggregation은 [Metric Reference](../reference/02-metrics.md)에서 설명한다.

## 이 문서 묶음의 범위

각 model 문서는 현재 ver3 source가 실제로 수행하는 동작을 기준으로 작성한다. 일반적인 논문의 모든
variant를 설명하지 않으며, 현재 CLI로 전달되지 않는 constructor option은 code-level option으로
구분한다. 공통 contract는 [Model Contract](../architecture/01-model-contract.md), 조립 규칙은
[Model Assembly](../architecture/02-model-assembly.md)에서 설명한다.

핵심을 한 문장으로 정리하면, 모든 model은 같은 corner를 예측하지만 학습 과정에서 corner를 바라보는
표현이 다르다. model 문서를 읽을 때는 항상 `target`, `raw output`, `loss`, `postprocess`의 네 항목을
연결해서 확인한다.
