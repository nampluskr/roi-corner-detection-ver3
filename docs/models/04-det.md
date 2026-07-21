# Custom Corner Detection (`det`)

`det`는 corner를 좌표 vector나 mask가 아니라 class가 있는 작은 object처럼 취급한다. image를 coarse
grid로 나누고 각 corner가 어느 cell에 있는지 분류한 뒤, cell 내부의 세부 위치 offset을 회귀한다. 이
문서는 object detection의 기본 생각을 현재 custom `DetModel`의 grid target과 연결해 설명한다.

## 1. Corner를 object로 보는 관점

일반적인 object detector는 image에서 `class`, `bounding box`, `confidence`를 예측한다. corner는 실제
면적을 가진 object가 아니지만, 다음과 같이 작은 pseudo-object로 바꿀 수 있다.

| corner | class id | 위치 |
| --- | ---: | --- |
| `TL` | 0 | 좌상 corner coordinate |
| `TR` | 1 | 우상 corner coordinate |
| `BR` | 2 | 우하 corner coordinate |
| `BL` | 3 | 좌하 corner coordinate |

class를 분리하면 detector는 네 corner의 순서를 postprocess에서 다시 추정할 필요가 없다. class 0의
예측은 항상 `TL`로 해석한다.

## 2. Grid cell이 필요한 이유

normalized coordinate를 바로 회귀하는 대신 image를 `Gh x Gw` grid로 나눈다. 각 corner는 하나의 cell에
배정되고, network는 다음 두 문제를 함께 푼다.

1. 각 class의 corner가 어느 cell에 있는지 찾는다.
2. 선택된 cell 안에서 정확히 어느 위치에 있는지 offset을 예측한다.

기본 `image_size=224`, `grid_stride=16`이면 grid는 `14 x 14`다. 한 cell은 원본 image의 약 `16 x 16`
pixel 영역을 담당한다.

## 3. 입출력 계약

`det`가 사용하는 주요 tensor는 다음과 같다.

| 단계 | shape | 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | RGB input batch |
| corner target | `(B, 4, 2)` | normalized four corners |
| class target | `(B, 4, Gh, Gw)` | class별 positive cell |
| box target with `box` head | `(B, 4, Gh, Gw)` | `dx`, `dy`, `w`, `h` |
| box target with `point` head | `(B, 2, Gh, Gw)` | `dx`, `dy` |
| positive mask | `(B, 1, Gh, Gw)` | regression loss를 계산할 cell |
| raw output | dictionary | `cls` logit map과 `box` raw map |
| final output | `(B, 4, 2)` | class별 best cell을 decode한 corner |

`box`라는 dictionary key는 `point` head에서도 유지된다. 이 key는 regression branch를 의미하며 실제
channel 수는 head에 따라 4 또는 2다.

## 4. 전체 architecture

현재 custom detector의 흐름은 다음과 같다.

```text
images
-> stage-returning backbone
-> CNNBackboneAdapter
-> FeatureBundle.stages
-> MultiScaleNeck
-> DetectionHead
-> cls map + box or point map
```

`det`는 stage pyramid를 요구하므로 token-only ViT와 DINOv2 계열은 지원하지 않는다. custom CNN,
torchvision CNN, timm CNN 중 registry에 등록된 network를 사용한다.

## 5. Multi-scale neck

backbone의 얕은 stage는 위치 정보가 풍부하고 깊은 stage는 semantic context가 풍부하다. neck은 선택한
`grid_stride`부터 더 깊은 stage를 common channel 수로 projection하고 top-down 방식으로 합친다.

```text
deep stage -> upsample -> add lateral stage -> convolution -> repeat
```

기본 output channel은 256이다. `grid_stride`가 backbone의 `stage_strides`에 존재하지 않으면 model 생성 시
오류가 발생한다. upsample feature와 skip feature shape가 다를 때도 오류를 발생시켜 잘못된 pyramid
조합을 드러낸다.

## 6. Detection head

`DetectionHead`는 shared convolution trunk 뒤에서 두 branch로 나뉜다.

| branch | output | 책임 |
| --- | --- | --- |
| classification | 4 channel | cell마다 `TL`, `TR`, `BR`, `BL` score |
| regression | 4 또는 2 channel | cell 내부 offset과 선택적인 size |

classification은 class별 channel을 사용하지만 regression은 class-agnostic이다. 네 class가 같은 regression
map을 공유하며, 각 class가 선택한 cell 위치에서 offset을 읽는다.

## 7. Cell assignment

normalized corner `(x, y)`를 grid에 배정하는 식은 다음과 같다.

$$
g_x = \lfloor xG_w \rfloor, \qquad g_y = \lfloor yG_h \rfloor
$$

cell 내부 offset은 다음과 같다.

$$
d_x = xG_w - g_x, \qquad d_y = yG_h - g_y
$$

`dx`, `dy`는 0 이상 1 미만이다. 예를 들어 `x=0.62`, `Gw=14`라면 `xGw=8.68`이므로 cell index는 8,
cell 내부 offset은 0.68이다.

`DetPreprocessor`는 해당 class channel의 `(gy, gx)`만 1로 만들고 나머지는 0으로 둔다.

## 8. `box`와 `point` head

두 head의 차이는 regression target channel에 있다.

| head | channel | target | final corner에 사용되는 값 |
| --- | ---: | --- | --- |
| `box` | 4 | `dx`, `dy`, fixed `w`, fixed `h` | `dx`, `dy` |
| `point` | 2 | `dx`, `dy` | `dx`, `dy` |

`box` head의 width와 height는 실제 corner 크기가 아니다. corner를 detection 형식으로 학습하기 위해 둔
고정 pseudo-box size이며 custom `det`의 기본값은 normalized 0.1이다. 현재 postprocessor는 width와 height를
사용하지 않고 center만 corner로 복원한다.

따라서 두 head의 최종 coordinate decode는 같지만 training objective의 regression channel 수가 다르다.

## 9. Positive mask

regression loss는 모든 background cell에서 계산하면 안 된다. 정답 corner가 없는 cell에는 올바른 offset이
정의되지 않기 때문이다.

`DetPreprocessor`는 네 class target의 channel maximum을 취해 `(B, 1, Gh, Gw)` positive mask를 만든다.
적어도 한 corner class가 배정된 cell에서만 regression loss가 활성화된다.

## 10. Classification focal loss

grid의 대부분은 background이고 positive cell은 class마다 하나뿐이다. ordinary BCE를 사용하면 많은 easy
background가 loss를 지배할 수 있다. `FocalLoss`는 이미 잘 맞힌 example의 weight를 줄인다.

각 cell의 target class 값을 `y`, sigmoid probability를 `p`라고 하자. 정답 class에 대한 probability를
`p_t`로 묶으면 focal factor는 `(1-p_t)^gamma`다. 현재 기본 `gamma=2`, positive와 negative balance를 위한
`alpha=0.25`를 사용한다.

예측이 쉬운 background에서 `p_t`가 1에 가까우면 focal factor가 매우 작아진다. model은 상대적으로
어려운 positive cell과 false positive에 더 집중한다.

## 11. Masked Smooth L1 loss

regression branch의 첫 두 channel은 raw logit이므로 sigmoid를 적용해 `dx`, `dy`를 0과 1 사이로 만든다.
width와 height channel에는 sigmoid를 적용하지 않는다.

prediction과 target 차이에 positive mask를 곱한 뒤 Smooth L1을 계산한다. 작은 오차에는 quadratic,
큰 오차에는 linear penalty를 사용해 MSE보다 outlier에 덜 민감하게 만든다. denominator도 positive cell
수와 regression channel 수를 기준으로 계산한다.

## 12. 학습 흐름

custom detector의 한 batch 학습은 다음 순서다.

```text
images -> DetModel -> {cls logits, box raw}
corners -> DetPreprocessor -> {cls target, box target, pos_mask}
cls logits + cls target -> FocalLoss
box raw + box target + pos_mask -> SmoothL1Loss
raw output -> DetPostprocessor -> corners -> PolygonIoU
```

classification과 regression은 역할이 다르므로 loss 이름도 `cls`, `box`로 따로 기록된다.

## 13. Inference와 decode

`DetPostprocessor`는 NMS나 score threshold를 사용하지 않는다. class channel마다 가장 높은 cell을 무조건
하나 선택한다.

선택한 cell을 `(gx, gy)`, sigmoid offset을 `(dx, dy)`라고 하면 corner는 다음과 같다.

$$
x = \frac{g_x + d_x}{G_w}, \qquad y = \frac{g_y + d_y}{G_h}
$$

classification confidence가 매우 낮아도 argmax는 존재하므로 항상 네 corner가 나온다. 이 점은 output
shape를 안정적으로 만들지만, confidence가 없는 실패를 자동으로 표시하지 못한다.

## 14. 단계별 예시

`14 x 14` grid에서 `TR=(0.82, 0.21)`이라고 가정한다.

```text
x * 14 = 11.48 -> gx=11, dx=0.48
y * 14 =  2.94 -> gy= 2, dy=0.94
```

class target channel 1의 cell `(2, 11)`이 1이 된다. inference에서 channel 1의 best cell도 `(2, 11)`이고
offset이 `(0.50, 0.90)`이라면 복원 좌표는 약 `(0.821, 0.207)`이다.

## 15. Warmup과 optimizer

기본 warmup 1 epoch 동안 extractor를 freeze하고 neck과 head를 `1e-4`로 학습한다. 이후 extractor는
`1e-5`, 나머지는 `1e-4`를 사용한다. scheduler는 validation IoU가 정체되면 learning rate를 줄인다.

classification loss와 regression loss의 숫자만으로 final corner quality를 판단하지 않고 IoU와
prediction을 함께 확인한다.

## 16. 대표 실패 원인과 진단

주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| corner가 cell 중앙 부근에 고정 | offset branch가 학습되지 않음 | `dx`, `dy` sigmoid 분포 |
| class별 corner가 잘못된 사분면에 있음 | classification channel 혼동 | class map visualization |
| score가 낮아도 corner가 출력됨 | argmax에 threshold가 없음 | class maximum confidence |
| 두 corner가 같은 cell을 선택 | coarse grid 또는 class confusion | grid stride, best index |
| `box` loss만 크고 coordinate는 괜찮음 | pseudo width와 height 회귀 영향 | `point` head 비교 |
| ViT network 생성 실패 | stage capability 없음 | supported CNN registry |

## 17. 다른 detector와의 차이

custom `det`와 external detector의 차이는 다음과 같다.

| 항목 | custom `det` | `torchdet`, `yolo`, `detr` |
| --- | --- | --- |
| architecture | project neck와 head | library whole model |
| assignment | corner를 한 grid cell에 직접 배정 | library native assignment or matching |
| loss | project focal + Smooth L1 | native detector loss |
| postprocess | class별 grid argmax | box selection, NMS 또는 query selection |
| 조립 가능성 | backbone, neck, head 경계가 보임 | 내부 결합을 유지함 |

## 18. Code mapping

개념과 source의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| backbone, neck, head assembly | `src/models/det/model.py` |
| cell assignment와 target | `src/models/det/preprocessor.py` |
| class argmax와 offset decode | `src/models/det/postprocessor.py` |
| focal, Smooth L1, optimizer | `src/models/det/wrapper.py` |
| top-down feature fusion | `src/components/necks.py` |
| classification와 regression branch | `src/components/heads.py` |

## 19. 실행 예시

기본 box head는 다음과 같이 실행한다.

```bash
python scripts/train.py --model det --network custom --head box --save
```

point head와 pretrained CNN을 비교하려면 다음처럼 실행한다.

```bash
python scripts/train.py --model det --network resnet18 --head point --save
```

grid stride, neck channel, box size는 constructor option이며 현재 공통 CLI에는 노출되지 않는다.

## 20. 핵심 요약

`det`는 image를 grid로 나누고 class별 positive cell과 cell 내부 offset을 학습한다. focal loss는 sparse
classification을, masked Smooth L1은 positive cell의 regression을 담당한다. 추론에서는 class마다 best
cell을 선택해 offset으로 corner를 복원한다. `box`와 `point` head의 핵심 차이는 training regression
channel이며 최종 coordinate는 둘 다 center를 사용한다.
