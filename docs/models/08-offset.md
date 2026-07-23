# Canonical Offset Regression (`offset`)

`offset`은 image를 입력받아 고정된 canonical square의 네 vertex에 대한 offset 8개를 예측하는 model이다.
좌표를 절대값으로 직접 예측하는 `reg`와 달리, 미리 정한 기준 사각형에서 각 corner가 얼마나 벗어나는지를
예측한다. 이는 DeTone 계열의 4-point parametrization으로 homography와 정보량이 동등하며, 새 방법론을
비교하는 저비용 regression baseline 역할을 한다. 이 문서는 canonical square 개념부터 현재 `OffsetModel`,
`OffsetWrapper`가 offset을 학습하고 corner를 복원하는 과정까지 설명한다.

## 1. 이 model이 답하는 질문

`offset`이 풀려는 질문은 다음과 같다.

> 네 corner가 대략 어디에 있을지 미리 정한 기준 사각형을 두고, image feature에서 각 corner가 그 기준점에서
> 얼마나 이동했는지를 예측할 수 있는가?

`reg`는 corner 좌표 자체를 회귀한다. `offset`은 기준점 대비 변위를 회귀한다. 정답 corner가 화면 중앙
근처에 몰려 있을 때, 기준점을 화면 중앙 근처의 사각형으로 두면 network는 0에 가까운 작은 값만 학습하면
된다. 이 작은-변위 가정이 offset parametrization의 핵심 prior다.

## 2. Canonical square

기준 사각형은 image 안쪽으로 `MARGIN = 0.25`만큼 들어온 정사각형이며, corner 순서는 `TL`, `TR`, `BR`,
`BL`이다.

```text
CANONICAL_CORNERS = [
    [0.25, 0.25],
    [0.75, 0.25],
    [0.75, 0.75],
    [0.25, 0.75],
]
```

정답 corner를 `y`, canonical vertex를 `c`라고 하면 학습 target은 변위 `y - c`다. 예를 들어 정답 `TL`이
`(0.10, 0.20)`이면 canonical `TL`은 `(0.25, 0.25)`이므로 offset target은 `(-0.15, -0.05)`다. 네 corner
모두에 대해 이렇게 변위를 구하고 `(B, 8)`로 펼친다.

## 3. 입출력 계약

`offset`의 각 단계에서 사용하는 shape는 다음과 같다.

| 단계 | shape | 값의 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | 정규화된 RGB batch |
| corner target | `(B, 4, 2)` | normalized `TL`, `TR`, `BR`, `BL` |
| offset target | `(B, 8)` | corner target에서 canonical vertex를 뺀 변위 |
| raw output | `(B, 8)` | 범위 제한 전 offset logit |
| final output | `(B, 4, 2)` | alpha*tanh와 canonical 복원을 적용한 corner |

여기서 raw output은 아직 범위가 제한되지 않은 변위 예측이다. 뒤에서 설명하는 `alpha*tanh`가 이 값을
제한된 변위로 바꾼다.

## 4. 전체 architecture

현재 `OffsetModel`은 `reg`와 동일한 backbone, adapter, coordinate head를 조립한다.

```text
images
-> backbone
-> native features
-> CNN or transformer adapter
-> global_feature or spatial_feature
-> coordinate head
-> 8 raw offsets
```

구조가 `reg`와 같은 이유는 offset이 표현만 다른 좌표 회귀이기 때문이다. 다른 점은 target 생성,
postprocess, loss 경로이며 network 조립은 공유한다.

## 5. Network와 head 선택

`offset`은 `custom`, torchvision backbone, timm backbone을 지원하며 `--network`/`--net`으로 선택한다.
`--head`는 `spatial`과 `gap`을 지원한다. 기본값은 `spatial`이다. `spatial` head는 strided convolution으로
local spatial arrangement를 보존하므로, 작은 변위를 예측하는 offset prior와 잘 맞는다.

두 head의 동작은 `reg`와 동일하다. `gap`은 global feature에 하나의 linear layer를 적용하고, `spatial`은
두 번의 strided convolution, `4 x 4` adaptive pooling, linear projection을 사용한다. head 세부는
[01-reg.md](01-reg.md)의 head 절과 [Model Assembly](../architecture/02-model-assembly.md)를 참조한다.

## 6. Target 생성

`OffsetPreprocessor`는 `(B, 4, 2)` corner에서 `CANONICAL_CORNERS`를 빼고 `(B, 8)`로 reshape한다. corner
순서는 반드시 `TL`, `TR`, `BR`, `BL`이어야 한다. 순서가 뒤바뀌면 잘못된 canonical vertex와 비교하게 되어
offset이 무의미해진다.

## 7. `alpha*tanh` bounding

offset은 좌표 전체 범위가 아니라 canonical vertex 주변의 제한된 변위여야 한다. 이를 위해 raw output `z`에
`alpha*tanh`를 적용한다.

$$
\text{offset} = \alpha \cdot \tanh(z)
$$

여기서 `alpha = 0.25`다. `tanh`는 임의의 실수를 `-1`과 `1` 사이로 제한하므로 각 변위는 `-0.25`와 `0.25`
사이로 제한된다. canonical vertex가 image 안쪽으로 `0.25` 들어와 있으므로, 이 제한 안에서 복원된 corner는
자연스럽게 `[0, 1]` 근처에 머문다. `reg`가 `sigmoid`로 좌표를 직접 `[0, 1]`로 제한하는 것과 대응된다.

## 8. Loss 작동 원리

기본 loss는 offset 전용 `OffsetSmoothL1Loss`다. ver3의 공용 `SmoothL1Loss`는 detection 계열의 dict 입력을
전제로 하므로, plain `(B, 8)` offset tensor에는 offset package 안에 정의한 smooth L1을 사용한다.

loss 경로는 postprocess와 일치해야 한다. `compute_losses`는 먼저 target을 만들고 raw output에 동일한
`alpha*tanh`를 적용한 뒤 비교한다.

```text
target = preprocessor(corner)          # canonical 대비 변위
offset = alpha * tanh(raw output)      # postprocess와 동일한 bounding
loss = smooth_l1(offset, target)
```

변위 `p`와 target `t`의 원소별 오차를 `d = |p - t|`라고 하면 smooth L1은 다음과 같다.

$$
L(d) =
\begin{cases}
0.5 \, d^2 / \beta, & d < \beta \\
d - 0.5 \, \beta, & d \ge \beta
\end{cases}
$$

여기서 `beta = 1`이다. batch loss는 8개 offset과 batch sample 전체의 평균이다. wrapper는 이 loss로
backward를 수행하고, postprocess 이후 polygon IoU를 training metric으로 누적한다.

## 9. Warmup과 optimizer

optimizer와 warmup 구성은 `reg`와 동일하다. 기본 `warmup_epochs=1`이면 첫 phase에서 extractor를 freeze하고
head만 `1e-4`로 학습한다. warmup이 끝나면 extractor는 `1e-5`, head는 `1e-4`를 사용한다. scheduler는 IoU를
기준으로 하는 `ReduceLROnPlateau`다. `warmup_epochs=0`이면 전체 parameter를 처음부터 `1e-4`로 학습한다.

## 10. Inference와 postprocess

추론에서는 target과 loss가 없다. `OffsetPostprocessor`가 다음 세 동작을 수행한다.

1. `(B, 8)` raw output에 `alpha*tanh`를 적용해 제한된 변위를 만든다.
2. 변위를 `(B, 4, 2)`로 reshape하고 canonical vertex를 더한다.
3. 결과를 `clamp(0, 1)`로 image 범위에 맞춘다.

threshold, contour, NMS가 없으므로 항상 네 점을 반환한다. `reg`와 마찬가지로 네 점이 교차하거나 몰리는
것을 막는 geometric constraint는 postprocessor에 없다.

## 11. 단계별 예시

한 sample의 raw output이 단순화를 위해 다음과 같다고 가정한다.

```text
[-0.62, -0.20, 0.62, -0.29, 0.55, 0.29, -0.55, 0.20]
```

`alpha*tanh`를 적용하면 대략 다음 변위가 된다.

```text
[-0.14, -0.05, 0.14, -0.07, 0.13, 0.07, -0.13, 0.05]
```

`(4, 2)`로 reshape하고 canonical vertex를 더하면 대략 다음 corner가 된다.

```text
TL=(0.11, 0.20), TR=(0.89, 0.18), BR=(0.88, 0.82), BL=(0.12, 0.80)
```

이 예시는 network가 corner를 직접 출력하는 것이 아니라 canonical square 대비 작은 변위를 출력하고
postprocessor가 기준점을 더해 corner로 복원한다는 점을 보여 준다.

## 12. 대표 실패 원인과 진단

주요 failure mode와 확인 방법은 다음과 같다.

| 증상 | 가능한 원인 | 먼저 확인할 항목 |
| --- | --- | --- |
| 모든 점이 canonical square에 붙음 | offset이 0 근처에서 벗어나지 못함 | raw output 분포, loss 감소 여부 |
| corner가 canonical 범위를 넘어야 하는데 못 감 | 정답 corner 변위가 `alpha`보다 큼 | `MARGIN`, `ALPHA` 설정과 data 분포 |
| 좌우 또는 상하 corner가 바뀜 | label order 또는 transform 오류 | CSV order, flip transform |
| polygon 크기는 비슷하나 위치가 흔들림 | spatial 정보 부족 | `gap`과 `spatial` 비교 |
| pretrained network가 빠르게 악화됨 | backbone learning rate가 큼 | warmup과 parameter group |

offset은 corner가 화면 대부분을 채우는 경우 `alpha` 제한에 부딪힐 수 있다. 정답 corner 변위 분포가
canonical square에서 `alpha`를 넘는 표본이 많다면 `MARGIN`과 `ALPHA`를 함께 검토한다.

## 13. 다른 표현과의 비교

`offset`의 위치는 다음과 같이 정리할 수 있다.

| 비교 대상 | `offset`의 장점 | `offset`의 한계 |
| --- | --- | --- |
| `reg` | 작은 변위 prior로 초기 수렴이 안정적 | canonical square에서 크게 벗어나는 corner에 `alpha` 제한이 작용 |
| `seg` | target과 postprocess가 단순함 | ROI area supervision을 사용하지 않음 |
| `peak` | dense map memory가 필요 없음 | local peak evidence를 직접 보존하지 않음 |
| `gcn` | 반복 정제 없이 빠름 | corner 관계를 명시적으로 message passing하지 않음 |

## 14. Code mapping

개념과 source file의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| backbone과 head 조립, canonical 상수 | `src/models/offset/model.py` |
| canonical 대비 변위 target | `src/models/offset/preprocessor.py` |
| alpha*tanh와 canonical 복원 | `src/models/offset/postprocessor.py` |
| optimizer, offset smooth L1, IoU, compute_losses | `src/models/offset/wrapper.py` |
| `gap`, `spatial` head | `src/components/heads.py` |

## 15. 실행 예시

기본 custom network와 `spatial` head는 다음과 같이 학습한다.

```bash
python scripts/train.py --model offset --network custom --head spatial --save
```

`gap` head 비교는 다음과 같이 실행한다.

```bash
python scripts/train.py --model offset --network resnet18 --head gap --save
```

checkpoint 평가와 예측에서도 학습 시 사용한 `model`, `network`, `head`를 동일하게 전달해야 한다.

## 16. 핵심 요약

`offset`은 corner를 절대 좌표가 아니라 고정된 canonical square 대비 변위로 학습한다.
`OffsetPreprocessor`가 canonical vertex를 빼서 target을 만들고, network가 만든 8개 offset을
`alpha*tanh`로 제한한 뒤 canonical vertex를 더해 corner를 복원한다. 작은 변위 prior 덕분에 baseline으로
안정적이지만, canonical square에서 크게 벗어나는 corner에는 `alpha` 제한이 작용하므로 data 분포와
`MARGIN`, `ALPHA` 설정을 함께 확인한다.
