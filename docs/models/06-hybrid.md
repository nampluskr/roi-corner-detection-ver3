# Hybrid Mask and Geometry (`hybrid`)

`hybrid`는 neural network와 classical computer vision을 한 pipeline에 결합한다. neural network는 ROI의
binary mask를 예측하고, OpenCV 기반 postprocessor는 mask edge에서 line을 찾고 교차시켜 corner를
복원한다. line 기반 경로가 실패하면 contour 기반 경로로 fallback한다.

이 문서는 learned component와 deterministic geometry가 각각 무엇을 담당하는지, 여러 fallback이 어떤
순서로 동작하는지 설명한다.

## 1. 두 접근법을 결합하는 이유

neural network는 illumination, texture, blur 같은 복잡한 appearance 변화에서 ROI 영역을 찾는 데 강하다.
classical geometry는 line intersection, contour approximation처럼 명확한 규칙으로 polygon corner를
계산할 수 있다.

`hybrid`의 핵심 질문은 다음과 같다.

> ROI가 어디에 있는지는 network가 학습하고, quadrilateral의 네 변과 corner를 계산하는 일은 명시적인
> geometry에 맡기면 두 접근법의 장점을 함께 얻을 수 있는가?

이 방식은 corner decode의 원리를 추적하기 쉽지만, mask quality와 geometry parameter 두 부분 모두가
성공해야 한다.

## 2. 입출력 계약

`hybrid`의 neural part는 segmentation과 같은 shape를 사용한다.

| 단계 | shape | 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | RGB input batch |
| corner target | `(B, 4, 2)` | normalized quadrilateral |
| mask target | `(B, 1, Hm, Wm)` | filled polygon mask |
| raw output | `(B, 1, Hm, Wm)` | mask logits |
| probability map | `(B, Hm, Wm)` | sigmoid foreground probability |
| final output | `(B, 4, 2)` | geometry로 복원한 corner 또는 NaN |

최종 output이 NaN일 수 있다는 점이 simple `seg`와 중요한 차이다. postprocessor가 invalid geometry를
명시적으로 표현하므로 success rate가 의미 있는 metric이 된다.

## 3. Neural architecture

`HybridModel`은 stage-returning backbone, `CNNBackboneAdapter`, `UNetDecoder`, `MaskHead`를 사용한다.

```text
images -> backbone stages -> UNetDecoder -> MaskHead -> mask logits
```

구조는 custom `seg`와 유사하지만 기본 network는 `mobilenet_v3_large`다. lightweight pretrained encoder와
explicit geometry를 결합하는 기본 구성을 의도한다. `custom`, ResNet, EfficientNet, MobileNet, Swin, VGG,
timm CNN registry도 사용할 수 있다.

## 4. Target과 loss

`HybridPreprocessor`는 `SegPreprocessor`를 상속해 같은 polygon fill target을 만든다. 네 normalized corner를
mask coordinate로 바꾸고 내부를 1로 채운다.

기본 `BCEDiceLoss`는 BCE와 soft Dice를 하나의 loss object 안에서 더한다.

$$
L = L_{BCE} + \lambda(1 - Dice)
$$

현재 기본 `lambda=1`이다. BCE는 cell별 foreground와 background를 구분하고, Dice는 predicted foreground와
target foreground의 전체 overlap을 보완한다.

## 5. 학습과 추론의 경계

학습에서는 geometry postprocessor가 loss에 직접 참여하지 않는다.

```text
corners -> mask target
images -> mask logits
mask logits + mask target -> BCE-Dice loss
```

metric 계산과 inference에서만 mask logits를 geometry pipeline에 전달한다.

```text
mask logits -> probability and binary mask -> line path or contour fallback -> corners
```

line fitting과 contour approximation은 discrete operation이므로 현재 end-to-end gradient가 흐르지 않는다.
network는 mask quality만 loss로 학습한다.

## 6. Postprocess 전체 순서

`HybridPostprocessor`의 우선순위는 다음과 같다.

```text
mask logits
-> sigmoid probability map
-> threshold 0.5 binary mask
-> Canny edges
-> probabilistic Hough segments
-> top, bottom, left, right grouping
-> four fitted lines
-> adjacent intersections
-> cornerSubPix refinement
-> validate and order

if line path fails
-> largest contour
-> quadrilateral approximation
-> min-area rectangle fallback
-> extreme-point fallback

if all paths fail
-> NaN corners
```

이 순서를 이해하면 failure가 neural mask에서 시작했는지, line path에서 발생했는지, contour fallback에서도
복원하지 못했는지 구분할 수 있다.

## 7. Probability map과 binary mask

raw logit에 sigmoid를 적용해 float probability map을 만든다. threshold 0.5를 적용한 binary mask는 edge와
contour 검출에 사용한다. probability map 자체는 마지막 subpixel refinement에 사용한다.

postprocessor는 원본 RGB image를 받지 않는다. 따라서 `cornerSubPix`가 보는 intensity surface는 original
image gradient가 아니라 predicted probability map이다. 이 distinction은 refinement 결과를 해석할 때
중요하다.

## 8. Canny edge detection

Canny는 binary mask boundary에서 edge pixel을 찾는다. 현재 threshold는 low 50, high 150이다. binary mask에
255를 곱해 OpenCV input으로 전달한다.

mask가 깨끗한 quadrilateral이면 네 boundary를 따라 edge가 나타난다. mask가 조각나거나 작은 artifact가
많으면 불필요한 edge segment도 함께 생긴다.

## 9. Hough line segments

`HoughLinesP`는 edge pixel에서 line segment 후보를 찾는다. 현재 주요 parameter는 다음과 같다.

| parameter | 값 | 의미 |
| --- | ---: | --- |
| threshold | 20 | line candidate를 지지하는 최소 vote |
| min length fraction | 0.15 | mask size 대비 최소 segment 길이 |
| max gap | 10 | 하나의 segment로 연결할 최대 gap |

minimum length는 `min(mask height, mask width) * 0.15`로 계산한다. 너무 짧은 noise edge를 line으로
사용하지 않으려는 기준이다.

## 10. Side grouping

검출된 segment를 top, bottom, left, right 네 group으로 나눈다. 먼저 mask foreground의 centroid `(cx, cy)`를
구한다.

segment angle이 45도보다 작으면 horizontal 계열로 보고 midpoint가 centroid 위인지 아래인지에 따라 top,
bottom을 정한다. angle이 45도 이상이면 vertical 계열로 보고 midpoint가 centroid 왼쪽인지 오른쪽인지에
따라 left, right를 정한다.

이 규칙은 ROI가 대략 quadrilateral이고 네 side가 centroid를 둘러싼다는 가정에 기반한다. perspective가
매우 크거나 side angle이 ambiguous하면 grouping이 잘못될 수 있다.

## 11. Line fitting과 intersection

각 side group의 segment endpoint를 모아 `cv2.fitLine`으로 total least squares line을 fitting한다. line은
다음 implicit equation으로 표현한다.

$$
ax + by + c = 0
$$

corner는 adjacent side의 교점으로 만든다.

| corner | intersected sides |
| --- | --- |
| `TL` | top, left |
| `TR` | top, right |
| `BR` | bottom, right |
| `BL` | bottom, left |

두 line determinant의 절대값이 `1e-6`보다 작으면 near-parallel로 보고 line path를 실패 처리한다.

## 12. Subpixel refinement

line intersection은 float coordinate지만 segment와 binary edge에서 계산되므로 probability surface를 더
사용할 수 있다. intersection이 border에서 충분히 떨어져 있으면 `cv2.cornerSubPix`를 실행한다.

window size는 5, 최대 iteration은 40, epsilon은 0.001이다. point가 border에 너무 가까우면 valid window를
만들 수 없으므로 refinement를 건너뛰고 원래 intersection을 사용한다.

## 13. Ordering과 validity 검사

line path 결과는 mask width와 height로 나누어 normalized coordinate로 바꾼다. `order_corners`는 centroid
주변 angle로 정렬한 뒤 `x+y`가 가장 작은 point를 `TL`로 회전시킨다.

`is_invalid_corners`는 어떤 두 corner 사이의 normalized distance가 0.02보다 작으면 invalid로 판단한다.
line path가 네 점을 만들었더라도 duplicate에 가까우면 contour fallback으로 넘어간다.

## 14. Contour fallback

line path가 실패하면 binary mask의 external contour를 찾고 면적이 가장 큰 contour를 선택한다. 여러
artifact 중 주요 ROI를 선택하려는 규칙이다.

`approxPolyDP`는 contour perimeter에 여러 epsilon fraction을 순서대로 적용한다. 정확히 네 vertex를 얻으면
이를 사용한다. 네 점을 얻지 못하면 `minAreaRect`와 `boxPoints`로 최소 회전 rectangle을 만든다.

이 fallback은 line grouping보다 다양한 contour를 처리하지만, min-area rectangle은 perspective
quadrilateral을 rectangle에 가깝게 단순화할 수 있다.

## 15. Extreme-point와 NaN fallback

contour 결과가 invalid하면 simple `mask_to_corners`를 사용한다. foreground의 `x+y`, `x-y` 극점으로 네
corner를 만든다. 이 결과도 invalid하거나 contour 자체가 없으면 `(4, 2)` 전체를 NaN으로 반환한다.

NaN은 조용히 중앙 좌표를 반환하는 대신 geometry failure를 명시한다. `SuccessRate`는 finite prediction의
비율을 계산하므로 이 실패를 집계할 수 있다.

## 16. 단계별 예시

깨끗한 trapezoid mask가 있다고 가정한다. Canny가 네 boundary edge를 찾고 Hough가 각 boundary에서 여러
segment를 만든다. centroid 위의 horizontal segment는 top, 아래는 bottom으로 group된다. 각 group의 모든
endpoint를 line fitting한 뒤 top-left 교점이 `TL`이 된다.

top edge가 끊겨 Hough line이 부족하면 line path는 `None`을 반환한다. 그러면 largest contour를
approximation해 네 점을 찾는다. 이처럼 fallback은 앞 단계의 일부 결과를 억지로 사용하지 않고 별도
경로로 다시 복원한다.

## 17. Warmup과 optimizer

기본 warmup 1 epoch 동안 extractor를 freeze하고 decoder와 mask head를 `1e-4`로 학습한다. 이후 extractor는
`1e-5`, 나머지는 `1e-4`를 사용한다. 기본 backbone은 pretrained `mobilenet_v3_large`다.

wrapper metric은 polygon IoU와 success rate다. mask loss가 감소해도 geometry failure가 남을 수 있으므로
두 metric을 함께 본다.

## 18. 대표 실패 원인과 진단

주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| NaN corner | 빈 mask 또는 모든 geometry fallback 실패 | binary mask, contour count |
| success rate는 높지만 IoU가 낮음 | fallback이 finite하지만 부정확한 quad 생성 | path별 output 비교 |
| top과 side가 바뀜 | angle 또는 centroid grouping 오류 | Hough segment visualization |
| corner가 rectangle 형태로만 나옴 | `minAreaRect` fallback 사용 | approx vertex count |
| line corner가 border에서 거침 | subpixel window 조건 불충족 | intersection margin |
| 작은 돌출부가 corner를 당김 | contour 또는 extreme point 영향 | largest contour와 cleaned mask |
| mask는 좋아 보이지만 line path 실패 | Hough parameter 또는 broken edge | Canny and Hough output |

## 19. `seg`와 비교

두 model은 mask network를 공유하지만 postprocess가 다르다.

| 항목 | `seg` | `hybrid` |
| --- | --- | --- |
| target | binary polygon mask | binary polygon mask |
| neural output | mask logits | mask logits |
| default loss | separate BCE and Dice | combined BCE-Dice |
| corner decode | mask extreme points | line path, contour, extreme fallback |
| failure output | 빈 mask에서 zero corners | 최종 실패에서 NaN corners |
| 추가 dependency | NumPy geometry | OpenCV geometry |

## 20. Code mapping

개념과 source의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| mask model assembly | `src/models/hybrid/model.py` |
| segmentation target reuse | `src/models/hybrid/preprocessor.py` |
| Canny, Hough, fitting, fallback | `src/models/hybrid/postprocessor.py` |
| BCE-Dice, IoU, success rate | `src/models/hybrid/wrapper.py` |
| ordering과 validity | `src/utils/geometry.py` |

## 21. 실행 예시

기본 MobileNetV3 hybrid는 다음과 같이 실행한다.

```bash
python scripts/train.py --model hybrid --network mobilenet_v3_large --head hybrid --save
```

custom backbone 비교는 다음과 같다.

```bash
python scripts/train.py --model hybrid --network custom --head hybrid --save
```

OpenCV postprocess parameter는 현재 공통 CLI에 노출되지 않고 source constant와 constructor default로
관리된다.

## 22. 핵심 요약

`hybrid`는 network가 binary mask를 학습하고 classical pipeline이 corner geometry를 복원한다. 우선
Canny, Hough, side grouping, line intersection, probability-map refinement를 사용하고, 실패하면 contour,
minimum rectangle, extreme point 순으로 fallback한다. 모든 경로가 실패하면 NaN을 반환한다. 이 model은
mask quality뿐 아니라 어느 geometry path가 선택되었는지까지 추적해야 올바르게 진단할 수 있다.
