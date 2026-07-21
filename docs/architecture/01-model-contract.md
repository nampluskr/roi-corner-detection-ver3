# Model Contract

이 문서는 서로 다른 corner detection model이 같은 dataset, trainer, evaluator, predictor에서 동작하기 위해
지켜야 하는 공통 계약을 설명한다. 여기서 계약은 특정 class의 구현 방법이 아니라 component 사이에서
주고받는 data의 shape, 값의 의미, 순서와 책임을 뜻한다.

처음 project를 읽는 사용자는 model architecture보다 이 계약을 먼저 이해하는 것이 좋다. `reg`는 좌표를
출력하고 `seg`는 mask를 출력하지만, 최종적으로 같은 corner contract로 돌아오기 때문에 하나의 평가
pipeline에서 비교할 수 있다.

## 1. 계약이 필요한 이유

model마다 native output을 그대로 사용하면 trainer와 evaluator가 model 수만큼 분기해야 한다. 예를 들어
coordinate tensor, segmentation mask, torchvision detection dictionary, DETR output object를 evaluator가
모두 직접 이해해야 한다.

이 project는 model package 안에서 native 표현을 처리하고, package 바깥에서는 다음 공통 기준을 사용한다.

```text
common image and corner target
-> model-specific training representation
-> model-specific raw output
-> common final corners
```

이 경계 덕분에 새로운 model을 추가해도 dataset과 standalone evaluator를 다시 만들 필요가 없다.

## 2. Image input 계약

기본 input은 RGB image batch다.

| 항목 | 기본 계약 |
| --- | --- |
| tensor shape | `(B, 3, H, W)` |
| channel order | RGB |
| dtype | floating point tensor |
| 기본 size | `H = W = 224` |
| value transform | tensor 변환 후 ImageNet mean과 standard deviation으로 normalize |

`B`는 batch size다. `3`은 red, green, blue channel 수다. `H`, `W`는 network에 들어가는 resized image의
height와 width다.

ImageNet normalization은 channel마다 다음 계산을 적용한다.

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

현재 mean은 `[0.485, 0.456, 0.406]`, standard deviation은 `[0.229, 0.224, 0.225]`다. local pretrained
backbone이 기대하는 input scale과 맞추기 위한 기준이다.

## 3. Corner coordinate 계약

정답과 final prediction은 `(B, 4, 2)` shape를 사용한다. 한 sample은 네 point, 한 point는 `(x, y)` 두
coordinate를 가진다.

corner 순서는 다음과 같이 고정한다.

| index | 이름 | 의미 |
| ---: | --- | --- |
| 0 | `TL` | top-left |
| 1 | `TR` | top-right |
| 2 | `BR` | bottom-right |
| 3 | `BL` | bottom-left |

순서는 polygon edge의 연결 관계이기도 하다. `TL`, `TR`, `BR`, `BL`을 차례대로 연결하면 quadrilateral
boundary가 된다. 같은 네 점이라도 순서가 다르면 coordinate loss와 polygon geometry가 달라진다.

## 4. Normalized coordinate

coordinate는 pixel index가 아니라 `[0, 1]` normalized 값이다. image width를 `W`, height를 `H`, pixel
coordinate를 `(u, v)`라고 하면 개념적으로 다음 관계를 사용한다.

$$
x = \frac{u}{W}, \qquad y = \frac{v}{H}
$$

예를 들어 `224 x 224` image에서 normalized `(0.25, 0.75)`는 대략 pixel `(56, 168)` 위치다. normalized
coordinate를 사용하면 image resize 후에도 label을 다시 pixel 단위로 조정할 필요가 없다.

현재 transform의 `Resize`가 corner를 변경하지 않는 이유도 corner가 이미 normalized이기 때문이다.

## 5. Corner order와 augmentation

geometric augmentation은 image와 coordinate를 함께 바꿔야 한다. horizontal flip은 `x`를 `1-x`로 바꾸는
것만으로 끝나지 않는다. 원래 `TR`이 flip 후 새 `TL`이 되므로 index도 재배열해야 한다.

현재 horizontal, vertical flip은 좌표 변환과 ordering을 함께 처리한다. rotation은 pixel space에서 네
point를 회전하고, 어느 point라도 `[0, 1]` 밖으로 나가면 해당 augmentation을 건너뛴다.

image만 변환하고 corner를 그대로 두면 network는 서로 다른 위치를 정답으로 학습한다. 이는 loss가
감소하지 않거나 좌우 corner가 뒤바뀌는 대표 원인이다.

## 6. Target, raw output, final output

초보자가 가장 자주 혼동하는 세 표현은 다음과 같다.

| 용어 | 생성 주체 | 사용 시점 | 예시 |
| --- | --- | --- | --- |
| corner target | dataset | 모든 labeled batch | `(B, 4, 2)` |
| model-specific target | preprocessor | training과 validation loss | mask, Gaussian map, pseudo-box |
| raw output | model forward | loss와 postprocess 이전 | logit, dictionary, native object |
| final output | postprocessor | metric과 prediction 저장 | `(B, 4, 2)` |

raw output을 곧바로 corner라고 가정하면 안 된다. `seg` raw output은 mask logit이고, `gcn` raw output은
모든 refinement step을 포함하며, detector는 box와 score를 포함한다.

## 7. Model package의 네 component

model package는 책임을 네 부분으로 나눈다.

### 7.1 Model

`Model`은 image에서 raw output을 계산하는 neural network다. training target이나 output directory를 알지
않는다. 일반적인 signature는 `raw_output = model(images)`지만 external detector는 native label을 함께
받는 경우가 있다.

### 7.2 Preprocessor

`Preprocessor`는 common corner target을 training representation으로 바꾼다. 예를 들어 `seg`는 polygon을
mask로 rasterize하고 `det`는 corner를 grid cell에 assignment한다. inference에는 정답이 없으므로
preprocessor를 사용하지 않는다.

### 7.3 Postprocessor

`Postprocessor`는 raw output을 common final corner로 decode한다. sigmoid와 reshape만 할 수도 있고,
threshold, NMS, contour, line intersection을 사용할 수도 있다. metric은 postprocess 이후 결과를 본다.

### 7.4 Wrapper

`Wrapper`는 model, preprocessor, postprocessor를 묶고 device, optimizer, scheduler, loss, metric과
train/eval/predict step을 관리한다. external whole-model이 native training interface를 요구하면 wrapper가
공통 step을 override한다.

## 8. 학습 계약

기본 `BaseWrapper.train_step`은 다음 순서를 따른다.

```text
move images and corners to device
-> model forward
-> preprocess corners
-> compute named losses
-> weighted loss sum
-> backward
-> optimizer step
-> postprocess raw output
-> update metrics
```

loss는 model-specific representation을 비교하고 metric은 common corner를 비교한다. 따라서 loss 숫자는
model마다 의미와 scale이 달라질 수 있지만 IoU와 corner distance는 공통 기준으로 비교할 수 있다.

shared 계산의 수식과 aggregation은 [Loss Reference](../reference/01-losses.md)와
[Metric Reference](../reference/02-metrics.md)에서 설명한다.

## 9. Inference 계약

inference에는 target, preprocessor, loss가 없다.

```text
images -> model -> raw output -> postprocessor -> (B, 4, 2)
```

`Wrapper.predict_step`은 final corner를 CPU NumPy array로 반환한다. `Evaluator`와 `Predictor`는 model
종류를 모르고 이 array만 사용한다.

## 10. Feature 계약

composable model은 backbone native output을 바로 consumer에 전달하지 않고 `FeatureBundle`로 변환한다.

| field | 일반 shape | 의미 | 주요 consumer |
| --- | --- | --- | --- |
| `global_feature` | `(B, C)` | image 전체를 요약한 vector | `reg` gap head |
| `spatial_feature` | `(B, C, Hf, Wf)` | 위치가 남은 final map | `reg` spatial, `gcn` |
| `stages` | feature map list | 여러 resolution의 encoder output | decoder와 neck |

CNN adapter는 final map을 global average pooling해 global feature를 만들 수 있다. Transformer adapter는
class token을 global feature로, patch token을 2D spatial map으로 reshape할 수 있다.

## 11. Feature capability와 `FeatureSpec`

모든 backbone이 모든 field를 제공하지는 않는다. `FeatureSpec`은 channel, stride, stage metadata를 기록하고
consumer가 필요한 capability를 확인하게 한다.

예를 들어 U-Net decoder는 `stages`가 필요하다. token-only ViT adapter는 stage pyramid를 제공하지 않으므로
현재 `seg`, `peak`, `ridge`, `det`에 그대로 연결할 수 없다. `spec.require("stages")`는 이 조합을 model
생성 시점에 거부한다.

## 12. Loss와 metric의 상태 계약

shared loss와 metric은 `reset`, `update`, `compute` lifecycle을 가진다. batch마다 값을 누적하고 epoch가
끝나면 running mean을 반환한다.

trainer는 train과 validation 시작 전에 state를 reset한다. standalone evaluator도 test 시작 전에 fresh
metric state를 만든다. state를 reset하지 않으면 이전 epoch나 이전 dataset의 값이 섞인다.

## 13. Failure 표현

현재 postprocessor의 failure 표현은 완전히 하나로 통일되어 있지 않다.

| 계열 | 실패 시 가능한 결과 |
| --- | --- |
| `reg`, `gcn` | 항상 finite coordinate를 만들지만 geometry가 invalid할 수 있음 |
| simple `seg` | 빈 mask에서 zero corners |
| `torchdet`, `yolo` | class 후보가 없으면 `(0.5, 0.5)` fallback |
| `hybrid` | 모든 geometry path 실패 시 NaN corners |
| `detr` | class마다 항상 best query를 선택 |

`Predictor`의 current `success`는 모든 coordinate가 finite인지로 판단한다. 따라서 zero 또는 center fallback은
성공으로 기록될 수 있다. `failure_reason` column은 존재하지만 current predictor는 실패일 때 고정 문자열
`invalid_prediction`을 사용하고, 세부 postprocess 원인을 전달받지 않는다.

## 14. Metric에서 invalid sample

common `BaseMetric`은 NaN prediction을 평균에서 제외한다. `SuccessRate`는 finite 여부를 모든 sample에서
직접 센다. 따라서 IoU나 distance만 보면 NaN failure가 평균에서 빠져 결과가 좋아 보일 수 있다.

항상 success rate와 sample-level prediction을 함께 확인해야 한다. finite지만 잘못된 fallback은 success
rate에도 잡히지 않으므로 polygon IoU와 corner distance가 필요하다.

## 15. Image size에 대한 현재 제약

CLI parser는 `--image_size`를 dataset resize에 사용한다. 그러나 current `get_wrapper_kwargs`는
`image_size`를 wrapper constructor로 전달하지 않고 `network`, `head`, `warmup_epochs`만 전달한다.

따라서 공통 script에서 non-default `--image_size`를 지정하면 input은 새 크기로 resize되지만, dense target,
pseudo-box pixel conversion과 postprocessor는 wrapper default 224를 사용할 수 있다. shape mismatch 또는
잘못된 scale이 생길 수 있으므로 현재 standard CLI workflow에서는 `image_size=224`를 유지하는 것이
안전하다. 다른 size를 지원하려면 별도 code 변경과 검증이 필요하다.

## 16. Contract 위반의 대표 증상

자주 나타나는 문제는 다음과 같다.

| 증상 | 의심할 계약 |
| --- | --- |
| 좌우 corner가 교환됨 | corner order와 flip transform |
| loss shape mismatch | raw output과 preprocessor target resolution |
| checkpoint key mismatch | model, network, head assembly 불일치 |
| metric은 계산되지만 결과가 중앙에 몰림 | missing-class fallback |
| decoder 생성 실패 | stage capability와 stride metadata |
| non-default size에서 dense loss 실패 | CLI와 wrapper image size 불일치 |

## 17. Code mapping

계약의 source 위치는 다음과 같다.

| 계약 | 구현 |
| --- | --- |
| dataset corner shape | `src/data/dataset.py` |
| joint image-corner transform | `src/data/transforms.py` |
| feature bundle과 capability | `src/components/features.py` |
| base model contract | `src/models/base/model.py` |
| base target conversion | `src/models/base/preprocessor.py` |
| base output conversion | `src/models/base/postprocessor.py` |
| step lifecycle | `src/models/base/wrapper.py` |
| common evaluation | `src/core/evaluator.py` |
| prediction row | `src/core/predictor.py` |

## 18. 핵심 요약

모든 model은 `(B, 3, H, W)` image와 `(B, 4, 2)` normalized ordered corner를 공통 경계로 사용한다.
preprocessor는 corner를 model-specific target으로, model은 image를 raw output으로, postprocessor는 raw
output을 다시 common corner로 바꾼다. wrapper가 이 흐름과 optimizer lifecycle을 묶는다. 서로 다른 model을
비교할 수 있는 이유는 native 표현이 달라도 final corner contract가 같기 때문이다.
