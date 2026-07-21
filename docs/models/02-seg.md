# Segmentation Corner Recovery (`seg`)

`seg`는 corner 좌표를 직접 예측하지 않는다. 먼저 quadrilateral ROI 내부를 1, 외부를 0으로 나타내는
binary mask를 예측하고, mask의 네 극점에서 corner를 복원한다. 이 문서는 pixel classification의 의미,
U-Net 형태의 encoder-decoder, mask target과 loss, corner 복원 과정을 현재 구현에 맞춰 설명한다.

## 1. 좌표 대신 영역을 예측하는 이유

corner는 image에서 매우 작은 위치 정보다. 반면 ROI 내부에는 많은 pixel이 있고 boundary는 연속된
형태를 가진다. direct regression이 image 전체를 8개 숫자로 압축한다면 segmentation은 각 spatial
위치가 ROI에 속하는지를 학습한다.

이 접근의 핵심 질문은 다음과 같다.

> 네 corner를 직접 맞히는 것보다 ROI 영역 전체를 먼저 찾고 그 외곽에서 corner를 계산하는 것이 더
> 안정적인가?

ROI 내부와 외부의 texture 차이가 뚜렷하거나 boundary가 연속적으로 보일 때 segmentation이 유리할 수
있다. 반대로 mask가 둥글거나 끊기면 정확한 corner 복원이 어려울 수 있다.

## 2. Binary segmentation 기초

segmentation은 image의 각 pixel 또는 output cell을 분류한다. binary segmentation에서는 각 위치가
foreground인지 background인지 판단한다.

| target 값 | 의미 |
| ---: | --- |
| 0 | ROI 바깥 background |
| 1 | quadrilateral ROI 안 foreground |

network는 0이나 1을 직접 출력하지 않고 mask logit을 출력한다. sigmoid를 적용하면 각 위치의 foreground
probability가 된다. threshold가 0.5라면 probability가 0.5보다 큰 위치를 foreground mask로 선택한다.

## 3. 입출력 계약

`seg`의 tensor 흐름은 다음과 같다.

| 단계 | shape | 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | 정규화된 RGB batch |
| corner target | `(B, 4, 2)` | normalized quadrilateral |
| mask target | `(B, 1, Hm, Wm)` | rasterized binary mask |
| raw output | `(B, 1, Hm, Wm)` | mask logit |
| final output | `(B, 4, 2)` | mask에서 복원한 normalized corner |

`Hm`, `Wm`은 항상 원본 image size와 같지 않다. `SegWrapper`는 `image_size // mask_stride`로 target
resolution을 정한다. 기본 image size가 224일 때 custom backbone의 첫 stage stride가 2이면 mask는
`112 x 112`, ResNet의 첫 stage stride가 4이면 `56 x 56`이 된다.

## 4. 전체 architecture

현재 `SegModel`은 stage feature를 제공하는 CNN 계열 backbone, adapter, `UNetDecoder`, `MaskHead`를
조립한다.

```text
images
-> stage-returning backbone
-> CNNBackboneAdapter
-> FeatureBundle.stages
-> UNetDecoder
-> MaskHead
-> single-channel mask logits
```

Vision Transformer처럼 stage pyramid를 제공하지 않는 network는 현재 `seg`에서 지원하지 않는다. dense
mask를 복원하려면 서로 다른 resolution의 encoder feature가 필요하기 때문이다.

## 5. Encoder와 stage feature

encoder는 image resolution을 줄이면서 channel을 늘린다. resolution이 낮아질수록 넓은 영역의 문맥을
보지만 정확한 위치 정보는 거칠어진다. 예를 들어 custom backbone은 stride `(2, 4, 8, 16)`의 네 stage를
제공한다.

| stage | 상대 resolution | 역할 |
| --- | --- | --- |
| shallow stage | 높음 | edge와 위치 같은 세부 정보 |
| middle stage | 중간 | local shape와 texture |
| deep stage | 낮음 | ROI 전체의 semantic context |

`CNNBackboneAdapter`는 이 stage list를 `FeatureBundle.stages`에 담는다. `FeatureSpec`은 channel과 stride를
기록해 decoder가 올바른 shape를 조립할 수 있게 한다.

## 6. `UNetDecoder` 작동 원리

encoder의 가장 깊은 feature만 upsample하면 위치 정보가 부족할 수 있다. U-Net decoder는 깊은 feature를
한 단계씩 확대하면서 같은 resolution의 얕은 feature를 skip connection으로 다시 더한다.

```text
deepest stage
-> upsample
-> add shallower skip
-> convolution
-> repeat until first stage resolution
```

현재 `UNetDecoder`는 concatenation이 아니라 additive skip을 사용한다. upsample 결과와 skip feature의
spatial shape가 다르면 조용히 crop하지 않고 `ValueError`를 발생시킨다. 이 검사는 잘못된 backbone stage
조합이 학습 중에 묻히는 것을 막는다.

기본 upsample mode는 `interpolate_conv`다. interpolation으로 resolution을 늘린 뒤 convolution을 적용한다.
constructor 수준에서는 transposed convolution mode도 shared block을 통해 사용할 수 있지만 공통 CLI에는
별도 option으로 노출되지 않는다.

## 7. `MaskHead`

decoder는 아직 여러 channel의 feature를 출력한다. `MaskHead`는 `1 x 1` convolution으로 이를 하나의
mask logit channel로 projection한다.

`MaskHead`가 하는 일은 channel projection뿐이다. sigmoid, threshold, corner 복원은 model forward 안에
들어 있지 않다. 이 분리는 학습에서 raw logit을 안정적인 BCE loss에 직접 전달하고, 추론에서 threshold를
postprocessor가 관리하게 한다.

## 8. Mask target 생성

`SegPreprocessor`는 normalized corner를 mask pixel 좌표로 바꾸고 PIL polygon fill을 사용한다.

처리 순서는 다음과 같다.

1. `(x, y)`에 `mask_size`를 곱해 mask 좌표를 만든다.
2. `TL`, `TR`, `BR`, `BL`을 잇는 polygon을 그린다.
3. polygon 내부와 outline을 1로 채운다.
4. batch를 `(B, 1, Hm, Wm)` float tensor로 만들고 원래 device로 돌려보낸다.

예를 들어 `mask_size=56`이고 corner가 `(0.25, 0.5)`라면 target 위치는 대략 `(14, 28)`이다. 네 점 사이의
모든 내부 pixel이 positive가 되므로 point target보다 훨씬 많은 supervised 위치를 제공한다.

## 9. BCE loss

`BCELoss`는 각 cell의 binary classification을 평가한다. logit을 `z`, target을 `y`라고 하면
`BCEWithLogitsLoss`는 sigmoid와 cross entropy를 수치적으로 안정적인 한 연산으로 계산한다.

직관적으로 target이 1인 cell에서는 probability가 1에 가까워지도록, target이 0인 cell에서는 0에
가까워지도록 만든다. ROI가 image에서 차지하는 면적과 background 면적이 크게 다르면 pixel 수가 많은
영역이 loss에 더 큰 영향을 줄 수 있다.

## 10. Dice loss

Dice는 두 mask가 얼마나 겹치는지를 본다. probability mask를 `p`, target mask를 `y`라고 하면 기본적인
soft Dice coefficient는 다음 형태다.

$$
Dice = \frac{2 \sum p y + s}{\sum p + \sum y + s}
$$

`s`는 빈 mask나 작은 분모에서 계산을 안정화하는 smoothing 값이다. 현재 `DiceLoss`는 `1 - Dice`를
batch 평균으로 계산한다. BCE가 cell별 정답을 학습한다면 Dice는 foreground 전체 overlap을 보완한다.

`SegWrapper`는 `bce`와 `dice` 두 loss를 같은 기본 weight로 더한다.

## 11. 학습 흐름

한 batch의 학습 흐름은 다음과 같다.

```text
images -> SegModel -> mask logits
corners -> SegPreprocessor -> binary masks
mask logits + binary masks -> BCE + Dice
mask logits -> SegPostprocessor -> corners -> PolygonIoU
```

loss는 mask를 직접 비교하고 metric은 corner로 복원한 뒤 polygon IoU를 계산한다. 따라서 mask loss가
감소해도 corner geometry가 반드시 같은 비율로 좋아지는 것은 아니다.

## 12. Warmup과 optimizer

기본 warmup은 1 epoch다. 첫 phase에서는 extractor를 freeze하고 decoder와 head를 `1e-4`로 학습한다.
두 번째 phase에서는 extractor를 `1e-5`, 나머지 component를 `1e-4`로 학습한다. pretrained backbone을
사용할 때 새 decoder가 먼저 task에 적응하도록 하는 목적이다.

custom backbone도 같은 wrapper 흐름을 사용한다. from-scratch network에서는 freeze된 첫 epoch의 효과가
pretrained network와 다를 수 있으므로 실험 조건에 warmup 여부를 기록한다.

## 13. Inference와 corner 복원

`SegPostprocessor`는 다음 순서로 raw output을 corner로 바꾼다.

1. mask logit에 sigmoid를 적용한다.
2. probability가 0.5보다 큰 cell을 binary mask로 만든다.
3. foreground pixel의 `x+y`, `x-y` 극값을 사용해 `TL`, `TR`, `BR`, `BL`을 찾는다.
4. mask width와 height로 나누어 normalized corner를 반환한다.

극점 방식은 빠르고 deterministic하다. 그러나 mask boundary를 직선으로 fitting하지 않기 때문에 작은
돌출 pixel이나 찢어진 영역이 corner 위치를 크게 움직일 수 있다.

빈 mask에서는 shared geometry helper가 네 점을 모두 0으로 반환한다. 이 결과는 finite하므로 단순 success
rate만으로 실패를 잡지 못할 수 있다. prediction CSV와 polygon IoU를 함께 확인해야 한다.

## 14. 단계별 예시

정답 quadrilateral이 mask target의 중앙 영역을 채운다고 가정한다. 학습 초기에는 logit이 모두 0 근처라
sigmoid probability가 약 0.5다. BCE와 Dice가 ROI 내부 probability를 높이고 외부를 낮춘다.

학습이 진행되어 threshold 후 mask가 다음처럼 만들어지면 postprocessor는 네 방향의 극점을 선택한다.

```text
background background background
background ROI ROI background
background ROI ROI background
background background background
```

실제 mask는 2D grid지만 원리는 같다. ROI의 좌상, 우상, 우하, 좌하 방향에서 가장 먼 foreground 위치가
corner가 된다.

## 15. 대표 실패 원인과 진단

주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| mask가 전부 background | class imbalance, underfitting | sigmoid 최대값, Dice, target mask |
| mask가 image 전체를 채움 | BCE balance 또는 threshold 문제 | probability histogram |
| corner 하나가 튀어나감 | 작은 foreground artifact | binary mask visualization |
| mask는 부드럽지만 corner가 둥글게 안쪽으로 들어감 | extreme point가 직선 교점을 복원하지 못함 | `hybrid`와 비교 |
| decoder shape 오류 | backbone stage stride 불일치 | `FeatureSpec`, stage shape |
| backbone 변경 후 target size 오류 | 첫 stage stride 변화 | `mask_stride`, raw output shape |

mask visualization은 segmentation model 진단에서 가장 중요한 중간 산출물이다. scalar loss만으로는
foreground가 올바른 위치에 형성되는지 알기 어렵다.

## 16. 다른 표현과의 비교

`seg`의 특성은 다음과 같다.

| 비교 대상 | `seg`의 장점 | `seg`의 한계 |
| --- | --- | --- |
| `reg` | 영역 전체의 spatial supervision | mask 생성과 geometry decode 필요 |
| `peak` | ROI 내부와 boundary 정보를 함께 사용 | corner별 channel을 직접 분리하지 않음 |
| `ridge` | 하나의 안정적인 foreground region | edge line을 명시적으로 fitting하지 않음 |
| `hybrid` | postprocess가 단순하고 빠름 | contour와 line fallback이 없음 |
| `torchseg` | backbone과 decoder를 project에서 조립 가능 | mature whole-model architecture를 그대로 쓰지 않음 |

## 17. Code mapping

개념과 source의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| backbone, adapter, decoder, head 조립 | `src/models/seg/model.py` |
| polygon mask rasterization | `src/models/seg/preprocessor.py` |
| threshold와 extreme-point decode | `src/models/seg/postprocessor.py` |
| BCE, Dice, optimizer, IoU | `src/models/seg/wrapper.py` |
| additive skip decoder | `src/components/decoders.py` |
| binary mask head | `src/components/heads.py` |
| mask extreme geometry | `src/utils/geometry.py` |

## 18. 실행 예시

기본 custom segmentation은 다음과 같이 실행한다.

```bash
python scripts/train.py --model seg --network custom --head mask --save
```

pretrained ResNet stage를 사용하는 비교 예시는 다음과 같다.

```bash
python scripts/train.py --model seg --network resnet18 --head mask --save
```

`seg`의 head는 현재 `mask` 하나다. `--head mask`는 공통 CLI 조립 형식을 유지하기 위한 명시값이다.

## 19. 핵심 요약

`seg`는 네 corner를 binary polygon mask로 바꾸고, encoder와 `UNetDecoder`가 mask logit을 예측하게 한다.
BCE는 cell별 분류를, Dice는 foreground overlap을 학습한다. 추론에서는 0.5 threshold와 mask 극점을 통해
corner를 복원한다. 이 model을 이해할 때는 mask resolution, foreground quality, postprocess가 corner
정확도에 미치는 영향을 함께 봐야 한다.
