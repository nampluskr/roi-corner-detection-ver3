# Direct Coordinate Regression (`reg`)

`reg`는 image를 입력받아 네 corner의 좌표 8개를 직접 예측하는 model이다. 중간에 mask, heatmap,
detection box를 만들지 않으므로 전체 흐름이 가장 단순하다. 이 문서는 regression이 무엇인지부터 현재
`RegModel`, `RegWrapper`가 좌표를 학습하고 복원하는 과정까지 설명한다.

## 1. 이 model이 답하는 질문

`reg`가 풀려는 질문은 다음과 같다.

> image 전체의 visual feature를 하나의 표현으로 압축한 뒤, 그 표현에서 네 corner 좌표를 바로 계산할 수
> 있는가?

분류는 입력이 어느 category에 속하는지 예측한다. 회귀는 연속적인 숫자를 예측한다. corner의 `x`, `y`
좌표는 0과 1 사이의 연속값이므로 이 문제는 회귀 문제로 볼 수 있다.

예를 들어 한 image의 정답이 다음과 같다고 가정한다.

```text
TL=(0.10, 0.20), TR=(0.90, 0.18), BR=(0.88, 0.82), BL=(0.12, 0.80)
```

`RegPreprocessor`는 이를 다음 8개 값으로 펼친다.

```text
[0.10, 0.20, 0.90, 0.18, 0.88, 0.82, 0.12, 0.80]
```

따라서 network가 해결해야 하는 최종 task는 image마다 8개의 연속값을 출력하는 것이다.

## 2. 입출력 계약

`reg`의 각 단계에서 사용하는 shape는 다음과 같다.

| 단계 | shape | 값의 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | 정규화된 RGB batch |
| corner target | `(B, 4, 2)` | normalized `TL`, `TR`, `BR`, `BL` |
| regression target | `(B, 8)` | corner target을 펼친 값 |
| raw output | `(B, 8)` | 범위 제한 전 logit |
| final output | `(B, 4, 2)` | sigmoid와 reshape를 적용한 corner |

여기서 logit은 아직 probability나 coordinate 범위로 제한되지 않은 network 출력이다. logit은 음수나
1보다 큰 값을 가질 수 있다. `sigmoid`는 임의의 실수를 0과 1 사이로 바꾸므로 normalized coordinate에
적합하다.

## 3. 전체 architecture

현재 `RegModel`은 backbone, adapter, coordinate head를 조립한다.

```text
images
-> backbone
-> native features
-> CNN or transformer adapter
-> global_feature or spatial_feature
-> coordinate head
-> 8 raw logits
```

backbone은 image에서 feature를 추출한다. adapter는 CNN feature map이나 transformer token처럼 서로 다른
native output을 `FeatureBundle`의 공통 field로 바꾼다. coordinate head는 선택한 field를 8개 logit으로
projection한다.

## 4. Network 선택

`reg`는 `custom`, torchvision backbone, timm backbone을 지원한다. CNN과 Vision Transformer 계열을
모두 사용할 수 있는 이유는 `reg`가 decoder stage를 요구하지 않고 global 또는 spatial final feature만
필요로 하기 때문이다.

현재 source에서 구분하는 network 계열은 다음과 같다.

| 계열 | 예시 | adapter가 사용하는 정보 |
| --- | --- | --- |
| custom CNN | `custom` | final feature map |
| torchvision CNN | `resnet18`, `efficientnet_b0`, `vgg16` | final feature map |
| torchvision transformer | `vit_b_16` | class token 또는 patch token |
| timm CNN | `wide_resnet50_2.tv_in1k` | final feature map |
| timm transformer | DeiT, CaiT registry 이름 | class token 또는 patch token |

pretrained network는 source에 등록된 local weight를 사용한다. registry에 없는 이름이나 local weight가 없는
경우 model 생성 단계에서 오류가 발생한다.

## 5. `gap`과 `spatial` head

`--head`는 feature를 좌표로 바꾸는 방식을 선택한다.

### 5.1 `gap` head

`gap`은 global feature를 사용한다. CNN에서는 final feature map에 global average pooling을 적용하고,
transformer에서는 class token을 사용한다. `CoordGapHead`는 dropout 뒤에 하나의 linear layer를 적용한다.

```text
feature map -> global average pooling -> vector -> dropout -> linear(8)
```

장점은 parameter와 계산이 적고 구조가 단순하다는 점이다. 단점은 spatial 위치 정보가 하나의 vector로
강하게 압축된다는 점이다.

### 5.2 `spatial` head

`spatial`은 위치가 남아 있는 feature map을 사용한다. CNN은 final feature map을 그대로 전달하고,
transformer는 patch token을 다시 2D grid로 reshape한다. `CoordSpatialHead`는 두 번의 strided convolution,
`4 x 4` adaptive pooling, dropout, linear projection을 사용한다.

```text
spatial feature -> strided conv -> strided conv -> 4x4 pooling -> flatten -> linear(8)
```

이 head도 마지막에는 8개 값으로 압축되지만, pooling 전에 local pattern을 추가 convolution으로 처리한다.
corner 위치에 spatial arrangement가 중요하다면 `gap`과 비교할 가치가 있다.

## 6. Target 생성

`RegPreprocessor`는 값을 변형하지 않고 `(B, 4, 2)`를 `(B, 8)`로 reshape한다. pixel coordinate로
바꾸거나 Gaussian을 그리지 않는다. 이 단순함은 direct regression의 중요한 장점이다.

corner 순서는 반드시 `TL`, `TR`, `BR`, `BL`이어야 한다. 순서가 뒤바뀌면 같은 polygon 모양이라도 loss는
서로 다른 위치의 값을 비교한다. transform 단계가 flip 후 corner 순서를 다시 정렬하는 이유도 여기에
있다.

## 7. Loss 작동 원리

기본 loss는 `WingLoss(apply_sigmoid=True)`다. 먼저 raw logit `z`에 sigmoid를 적용해 예측 좌표
`p = sigmoid(z)`를 만든다. 정답 좌표를 `y`라고 하면 원소별 오차는 `d = |p - y|`다.

Wing loss의 현재 형태는 다음과 같다.

$$
L(d) =
\begin{cases}
w \log(1 + d / \epsilon), & d < w \\
d - C, & d \ge w
\end{cases}
$$

여기서 `w = 10`, `epsilon = 2`이고 `C`는 두 구간이 이어지게 만드는 상수다. normalized coordinate의
오차는 최대 1이므로 현재 기본값에서는 사실상 logarithmic branch를 사용한다. 작은 오차 구간에서도
gradient가 남아 corner를 세밀하게 조정하도록 만든다.

batch loss는 8개 좌표와 batch sample 전체의 평균이다. wrapper는 이 loss로 backward를 수행하고,
postprocessor 이후 polygon IoU를 training metric으로 누적한다.

## 8. Warmup과 optimizer

기본 `warmup_epochs=1`이면 첫 phase에서 extractor를 freeze하고 head만 `1e-4` learning rate로 학습한다.
warmup이 끝나면 extractor는 `1e-5`, head는 `1e-4` learning rate를 사용한다. pretrained feature를 크게
흔들지 않으면서 새 coordinate head를 먼저 적응시키려는 구성이다.

`warmup_epochs=0`이면 전체 parameter를 처음부터 `1e-4`로 학습한다. `custom` backbone은 pretrained가
아니므로 짧은 warmup이 항상 유리하다고 단정할 수 없다. 비교할 때는 warmup 설정을 experiment 조건으로
기록한다.

## 9. Inference와 postprocess

추론에서는 target과 loss가 없다. `RegPostprocessor`가 다음 두 동작만 수행한다.

1. `(B, 8)` raw output에 sigmoid를 적용한다.
2. 결과를 `(B, 4, 2)`로 reshape한다.

별도의 threshold, contour, NMS가 없으므로 항상 네 점을 반환한다. 값도 0과 1 사이에 있다. 다만 네 점이
서로 교차하거나 같은 위치에 몰리는 것을 막는 geometric constraint는 현재 postprocessor에 없다.

## 10. 단계별 예시

한 sample의 raw logit이 단순화를 위해 다음과 같다고 가정한다.

```text
[-2.20, -1.39, 2.20, -1.52, 1.99, 1.52, -1.99, 1.39]
```

sigmoid를 적용하면 대략 다음 값이 된다.

```text
[0.10, 0.20, 0.90, 0.18, 0.88, 0.82, 0.12, 0.80]
```

reshape 후 첫 두 값은 `TL`, 다음 두 값은 `TR`이 된다. 이 예시는 network가 좌표를 직접 출력한다는 말이
실제로는 범위 제한 전 logit 8개를 출력하고 postprocessor가 좌표로 해석한다는 뜻임을 보여 준다.

## 11. 대표 실패 원인과 진단

주요 failure mode와 확인 방법은 다음과 같다.

| 증상 | 가능한 원인 | 먼저 확인할 항목 |
| --- | --- | --- |
| 모든 점이 중앙에 모임 | logit이 0 근처에서 벗어나지 못함 | raw output 분포, loss 감소 여부 |
| 좌우 또는 상하 corner가 바뀜 | label order 또는 transform 오류 | CSV order, flip transform |
| polygon 크기는 비슷하나 위치가 흔들림 | global feature의 spatial 정보 부족 | `gap`과 `spatial` 비교 |
| pretrained network가 빠르게 악화됨 | backbone learning rate가 큼 | warmup과 parameter group |
| IoU는 낮지만 좌표 오차는 작음 | 한 corner의 큰 오차 또는 polygon 교차 | `maxcd`, prediction CSV |

direct regression은 항상 finite corner를 만들기 쉬워 success rate만으로 품질을 판단하기 어렵다. 네 점의
ordering, polygon convexity, corner별 distance를 함께 확인한다.

## 12. 다른 표현과의 비교

`reg`의 위치는 다음과 같이 정리할 수 있다.

| 비교 대상 | `reg`의 장점 | `reg`의 한계 |
| --- | --- | --- |
| `seg` | target과 postprocess가 단순함 | ROI area supervision을 사용하지 않음 |
| `peak` | dense map memory가 필요 없음 | local peak evidence를 직접 보존하지 않음 |
| `det` | assignment와 NMS가 없음 | class별 spatial search가 없음 |
| `gcn` | 반복 정제 없이 빠름 | corner 관계를 명시적으로 message passing하지 않음 |

## 13. Code mapping

개념과 source file의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| backbone과 head 조립 | `src/models/reg/model.py` |
| target reshape | `src/models/reg/preprocessor.py` |
| sigmoid와 corner reshape | `src/models/reg/postprocessor.py` |
| optimizer, Wing loss, IoU | `src/models/reg/wrapper.py` |
| `gap`, `spatial` head | `src/components/heads.py` |

## 14. 실행 예시

기본 custom network와 `gap` head는 다음과 같이 학습한다.

```bash
python scripts/train.py --model reg --network custom --head gap --save
```

spatial head 비교는 다음과 같이 실행한다.

```bash
python scripts/train.py --model reg --network resnet18 --head spatial --save
```

checkpoint 평가와 예측에서도 학습 시 사용한 `model`, `network`, `head`를 동일하게 전달해야 한다.

## 15. 핵심 요약

`reg`는 `(B, 4, 2)` corner를 `(B, 8)` target으로 펼치고, network가 만든 8개 logit을 sigmoid로 좌표화한다.
구조와 postprocess가 가장 단순해 baseline으로 적합하다. 대신 spatial evidence와 polygon 관계를 하나의
coordinate vector 안에서 암묵적으로 학습해야 하므로, `gap`과 `spatial` head의 차이 및 ordering 오류를
중점적으로 확인한다.
