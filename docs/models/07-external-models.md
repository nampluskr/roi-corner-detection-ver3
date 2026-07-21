# External Whole Models (`torchseg`, `torchdet`, `yolo`, `detr`)

이 문서는 torchvision, Ultralytics, Hugging Face가 제공하는 complete model을 ROI corner task에 연결하는
방법을 설명한다. project가 backbone과 decoder를 직접 조립하는 model과 달리, external whole-model은
library가 설계한 내부 architecture와 native training rule을 가능한 한 유지한다.

현재 지원하는 model은 segmentation 계열 `torchseg`와 detection 계열 `torchdet`, `yolo`, `detr`다. 네
model의 공통점은 외부 model의 입출력을 project의 `(B, 4, 2)` corner contract로 바꾸는 adapter layer가
필요하다는 점이다.

## 1. Whole-model reuse란 무엇인가

backbone reuse는 pretrained encoder만 가져오고 decoder와 head는 project가 만든다. whole-model reuse는
encoder, neck or decoder, task head의 결합을 유지한 채 마지막 classifier와 입출력 interface만 task에
맞춘다.

두 방식의 차이는 다음과 같다.

| 항목 | composable project model | external whole-model |
| --- | --- | --- |
| 내부 component | project가 backbone, decoder, head를 조립 | library 조립을 유지 |
| loss | `src.components.losses` 사용 | library native loss 사용 가능 |
| output | project가 정한 tensor or dictionary | library native object |
| adaptation | feature adapter 중심 | classifier, target, wrapper, postprocessor 중심 |
| 장점 | component별 ablation이 쉬움 | 검증된 complete architecture를 재사용 |
| 부담 | task component를 직접 설계 | dependency와 native interface를 관리 |

## 2. 공통 project contract

external model도 입력 dataset과 최종 prediction은 project 계약을 따른다.

| 경계 | project 표현 |
| --- | --- |
| input image | `(B, 3, H, W)` normalized tensor |
| label source | `(B, 4, 2)` normalized `TL`, `TR`, `BR`, `BL` |
| final prediction | `(B, 4, 2)` normalized corner |
| evaluation | common Polygon IoU와 standalone metric bank |

차이는 이 공통 label을 external library target으로 바꾸는 preprocessor와 native output을 corner로 바꾸는
postprocessor에서 발생한다.

## 3. 현재 registry

지원 network는 source에 명시된 local weight registry를 따른다.

| model | supported network | 기본 network |
| --- | --- | --- |
| `torchseg` | `fcn_resnet50`, `deeplabv3_resnet50`, `deeplabv3_mobilenet_v3_large`, `lraspp_mobilenet_v3_large` | `fcn_resnet50` |
| `torchdet` | `fasterrcnn_resnet50_fpn`, `retinanet_resnet50_fpn`, `ssd300_vgg16` | `fasterrcnn_resnet50_fpn` |
| `yolo` | `yolov8n` | `yolov8n` |
| `detr` | `detr_resnet50` | `detr_resnet50` |

weight는 network download를 자동으로 수행하지 않고 `/mnt/d/backbones/` 아래 local file 또는 directory에서
읽는다. file이 없거나 optional package가 설치되지 않으면 model 생성 단계에서 명시적인 오류가 발생한다.

## 4. External `head`의 의미

detection external wrapper에서 `--head box`와 `--head point`는 library 내부 head architecture를 바꾸는
option이 아니다. 두 option은 corner를 pseudo-box로 만들 때 width와 height를 선택한다.

| head | normalized pseudo-box size |
| --- | ---: |
| `box` | 0.3 |
| `point` | 0.1 |

두 경우 모두 final prediction은 detected box의 center다. `point`라는 이름은 더 작은 pseudo-box로 corner를
점에 가깝게 표현한다는 뜻이다. `torchseg`의 `--head mask`도 공통 CLI compatibility를 위한 값이며
torchvision segmentation classifier 종류를 선택하지 않는다.

## 5. `torchseg`: complete segmentation model

`torchseg`는 torchvision segmentation architecture 전체를 사용한다. project `seg`가 stage-returning
backbone과 `UNetDecoder`를 조립하는 것과 달리 FCN, DeepLabV3, LR-ASPP의 내부 decoder를 그대로 유지한다.

### 5.1 Classifier replacement

COCO pretrained segmentation model은 여러 class channel을 출력한다. ROI task는 foreground와 background만
구분하면 되므로 마지막 classifier convolution을 output channel 1로 교체한다.

LR-ASPP는 low-resolution과 high-resolution classifier가 따로 있어 두 classifier를 모두 1 channel로
교체한다. 다른 model은 main classifier의 마지막 convolution과 존재하는 auxiliary classifier의 마지막
convolution을 교체한다.

forward는 torchvision output dictionary의 `"out"` tensor만 반환한다. auxiliary output은 current wrapper
loss에 사용하지 않는다.

### 5.2 Tensor contract

`TorchSegModel`은 output을 input image resolution으로 반환하므로 `mask_stride=1`이다. 기본 image size가
224라면 shape는 다음과 같다.

| 단계 | shape |
| --- | --- |
| image | `(B, 3, 224, 224)` |
| mask target | `(B, 1, 224, 224)` |
| raw output | `(B, 1, 224, 224)` |
| final corner | `(B, 4, 2)` |

`TorchSegPreprocessor`와 `TorchSegPostprocessor`는 custom `seg`의 mask rasterization과 extreme-point decode를
상속한다. 따라서 architecture는 다르지만 target, loss, postprocess는 직접 비교할 수 있다.

### 5.3 Loss와 training

기본 loss는 `BCELoss`와 `DiceLoss`다. `TorchSegWrapper`는 whole model 전체를 하나의 AdamW optimizer에
넣고 learning rate `1e-4`를 사용한다. constructor는 공통 signature로 `warmup_epochs`를 받지만 현재
구현은 phase warmup을 적용하지 않는다.

### 5.4 Inference

main output logit에 sigmoid와 threshold 0.5를 적용한 뒤 binary mask의 extreme point를 `TL`, `TR`, `BR`,
`BL`로 해석한다. 따라서 custom `seg`와 동일한 postprocess failure mode를 갖는다.

## 6. `torchdet`: torchvision detector reuse

`torchdet`는 Faster R-CNN, RetinaNet, SSD 전체를 사용한다. 세 architecture는 proposal과 assignment 방식이
다르지만 project wrapper는 공통 pseudo-box target과 class별 center decode로 통일한다.

### 6.1 Classifier replacement

pretrained COCO weight를 원래 class 수로 먼저 load한 뒤 task classifier를 교체한다.

| network | 교체 지점 |
| --- | --- |
| Faster R-CNN | ROI box predictor |
| RetinaNet | classification head |
| SSD | multi-scale classification head |

Faster R-CNN과 SSD는 background class가 별도 label 0이므로 corner label에 offset 1을 더한다. RetinaNet은
background를 별도 class channel로 두지 않아 offset 0을 사용한다.

### 6.2 Pseudo-box target

`TorchDetPreprocessor`는 normalized corner center를 pixel coordinate로 바꾼다. normalized box size를
`s`, image size를 `I`라고 하면 box edge는 다음과 같다.

$$
x_1 = xI - \frac{sI}{2}, \quad y_1 = yI - \frac{sI}{2}, \quad
x_2 = xI + \frac{sI}{2}, \quad y_2 = yI + \frac{sI}{2}
$$

sample마다 `{"boxes": boxes, "labels": labels}` dictionary를 만들고 image도 tensor batch가 아니라
tensor list로 detector에 전달한다.

### 6.3 Native loss

training mode에서 torchvision detector는 loss dictionary를 반환한다. Faster R-CNN이라면 classifier,
box regression, objectness, RPN box loss 등이 포함될 수 있다. wrapper는 dictionary의 모든 값을 더해
backward하고 각 항목을 이름별 running mean으로 기록한다.

evaluation mode에서는 detector가 box, label, score dictionary list를 반환한다. current `eval_step`은
native validation loss를 별도로 계산하지 않고 final corner IoU를 계산한다.

### 6.4 Class별 center decode

`TorchDetPostprocessor`는 corner class마다 해당 label의 detection만 고른다. 후보가 여러 개면 score가 가장
높은 box를 선택하고 center를 image size로 나눈다.

해당 class 후보가 하나도 없으면 corner의 초기값 `(0.5, 0.5)`가 유지된다. 이 값은 finite하므로 success
rate만으로 missing detection을 알기 어렵다.

### 6.5 Warmup

기본 warmup 1 epoch 동안 detector backbone을 freeze하고 나머지 parameter를 `1e-4`로 학습한다. 이후
backbone은 `1e-5`, 나머지는 `1e-4`를 사용한다.

## 7. `yolo`: Ultralytics YOLOv8 adaptation

`yolo`는 local `yolov8n.pt` checkpoint에서 complete model을 읽는다. Ultralytics inference checkpoint는
parameter의 `requires_grad`가 꺼져 있을 수 있으므로 current implementation은 전체 network를 다시 trainable
상태로 만든다.

### 7.1 Detection classifier replacement

YOLO detect module의 class prediction convolution을 4 class output으로 교체한다. class 이름은
`corner0`부터 `corner3`으로 설정하고 model metadata의 class 수도 4로 갱신한다. box regression branch와
distribution representation은 유지한다.

### 7.2 Native target batch

`YoloPreprocessor`는 sample마다 네 corner를 normalized `cx`, `cy`, `w`, `h` box로 만든다. 결과는 flat한
batch dictionary다.

| key | 의미 |
| --- | --- |
| `batch_idx` | 각 box가 어느 image에 속하는지 |
| `cls` | 0부터 3까지 corner class |
| `bboxes` | normalized `cx`, `cy`, `w`, `h` |
| `img` | wrapper가 추가하는 image batch |

torchvision detector처럼 sample별 dictionary list가 아니라 모든 target을 이어 붙이고 `batch_idx`로 sample을
구분한다.

### 7.3 Native loss

Ultralytics `v8DetectionLoss`를 그대로 사용한다. wrapper는 native total loss로 backward하고 detached loss
component를 `box`, `cls`, `dfl` 이름으로 누적한다.

| component | 직관적인 역할 |
| --- | --- |
| `box` | predicted box와 target box의 위치와 overlap |
| `cls` | corner class classification |
| `dfl` | discrete distribution으로 표현된 box edge regression |

이 loss의 scale은 custom `det`의 focal과 Smooth L1 scale과 같지 않으므로 숫자를 model 사이에서 직접
비교하지 않는다.

### 7.4 NMS와 corner decode

evaluation output은 Ultralytics decoded tensor다. `YoloPostprocessor`는 native NMS를 실행한다. 현재
default는 confidence threshold 0.001, IoU threshold 0.5, maximum detection 10이다.

NMS 이후 class별 최고 score box center를 corner로 사용한다. class 후보가 없으면 `(0.5, 0.5)`가 남는다.
낮은 confidence threshold는 후보 누락을 줄이지만 false positive가 들어올 가능성도 높인다.

### 7.5 Warmup

첫 phase에서는 final detect module을 제외한 backbone layer를 freeze하고 detection head를 `1e-4`로 학습한다.
다음 phase에서 backbone `1e-5`, 나머지 `1e-4`를 사용한다.

## 8. `detr`: transformer query detection

DETR은 convolutional feature와 transformer encoder-decoder를 사용해 fixed number의 object query를
예측한다. anchor와 NMS에 의존하는 detector와 달리 set prediction과 Hungarian matching을 사용하는 것이
핵심이다.

현재 implementation은 local Hugging Face snapshot에서 `DetrForObjectDetection`을 load하고 class mapping을
corner 4종으로 바꾼다. classifier size가 달라지는 것은 `ignore_mismatched_sizes=True`로 허용한다.

### 8.1 Query의 직관

각 query는 image에서 하나의 object 후보를 담당할 수 있는 learned slot이다. 모든 query가 class logit과
normalized box를 출력한다. training에서는 Hungarian algorithm이 target box와 query 사이의 one-to-one
matching을 찾는다.

이 matching은 target 순서대로 query index를 고정하지 않는다. 어떤 query가 `TL`을 담당할지는 sample마다
달라질 수 있으며 class label과 matching cost가 역할을 정한다.

### 8.2 DETR target

`DetrPreprocessor`는 sample마다 다음 dictionary를 만든다.

| key | shape | 의미 |
| --- | --- | --- |
| `class_labels` | `(4,)` | corner class 0, 1, 2, 3 |
| `boxes` | `(4, 4)` | normalized `cx`, `cy`, `w`, `h` |

box center는 corner coordinate이고 width와 height는 head에 따른 fixed pseudo-box size다. torchvision과
달리 pixel coordinate로 변환하지 않는다.

### 8.3 Native Hungarian loss

Hugging Face model은 label을 전달하면 total `loss`와 `loss_dict`를 반환한다. loss에는 classification,
box L1, generalized IoU 등 DETR native 항목이 포함될 수 있다. wrapper는 `output.loss`로 backward하고
`loss_dict`를 이름별로 기록한다.

gradient norm은 기본 1.0으로 clip한다. transformer fine-tuning에서 큰 gradient로 학습이 불안정해지는
것을 완화한다.

### 8.4 Query decode

`DetrPostprocessor`는 query class logit에 softmax를 적용한다. 각 corner class마다 그 class score가 가장
높은 query를 선택하고 query box의 center 두 값을 corner로 사용한다.

이 decode는 class별로 독립적인 argmax를 수행한다. 따라서 이론적으로 같은 query가 둘 이상의 class에서
선택될 수 있다. postprocessor는 NMS나 query uniqueness 재할당을 수행하지 않는다. box center는 `[0, 1]`
범위로 clamp한다.

### 8.5 Warmup과 optimizer

첫 phase에서는 backbone parameter를 제외하고 transformer와 classifier를 `1e-4`로 학습한다. 다음
phase에서는 backbone `1e-5`, transformer와 classifier `1e-4`를 사용한다. parameter name prefix로 backbone과
classifier group을 구분한다.

## 9. 학습 interface 비교

네 external model이 wrapper에서 받는 형태는 같지만 native 호출은 다음처럼 다르다.

| model | image 전달 | target 전달 | training output |
| --- | --- | --- | --- |
| `torchseg` | tensor batch | binary mask tensor | mask logits |
| `torchdet` | image tensor list | box dictionary list | loss dictionary |
| `yolo` | tensor batch | flat batch dictionary | native total and component loss |
| `detr` | `pixel_values` tensor | label dictionary list | model output with loss fields |

이 차이를 `Wrapper`가 흡수하기 때문에 `Trainer`는 model 이름과 관계없이 `train_step`을 호출할 수 있다.

## 10. Postprocess 비교

final corner를 만드는 방식은 다음과 같다.

| model | candidate 생성 | class별 선택 | missing class 처리 |
| --- | --- | --- | --- |
| `torchseg` | threshold mask | extreme geometry | 빈 mask에서 zero corners |
| `torchdet` | native detections | max score box | center `(0.5, 0.5)` |
| `yolo` | decoded boxes와 NMS | max score box | center `(0.5, 0.5)` |
| `detr` | fixed queries | max class score query | 항상 query 선택 |

`success`가 finite 여부만 검사한다면 center fallback이나 zero corner는 성공으로 기록될 수 있다. external
detector 결과는 class별 confidence와 prediction CSV를 함께 봐야 한다.

## 11. 단계별 pseudo-box 예시

`TR=(0.8, 0.2)`, `head=point`, box size 0.1이라고 가정한다.

YOLO와 DETR target은 normalized center-size 형식이다.

```text
[cx=0.8, cy=0.2, w=0.1, h=0.1]
```

image size 224인 torchvision target은 pixel edge 형식으로 바뀐다.

```text
center = (179.2, 44.8)
half size = 11.2
box = (168.0, 33.6, 190.4, 56.0)
```

세 detector의 native target 형식은 다르지만 final decode는 selected box center를 다시 `(0.8, 0.2)` 같은
normalized corner로 만드는 것이 목표다.

## 12. 대표 실패 원인과 진단

공통 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| model 생성 즉시 실패 | local weight 또는 package 없음 | registry path와 environment |
| checkpoint load mismatch | training과 evaluation network 다름 | CLI `model`, `network`, `head` |
| corner가 image 중앙에 반복 | class detection 누락 fallback | class별 candidate count |
| box는 검출되지만 corner class가 바뀜 | classifier adaptation 또는 label offset 오류 | label mapping |
| native loss는 감소하지만 IoU가 낮음 | pseudo-box objective와 center accuracy 차이 | center distance와 predictions |
| pretrained model이 빠르게 악화 | backbone learning rate 또는 warmup | parameter group과 epoch |

model별 추가 점검 항목은 다음과 같다.

| model | 추가 점검 |
| --- | --- |
| `torchseg` | output resolution, binary classifier, mask threshold |
| `torchdet` | background label offset, box coordinate unit |
| `yolo` | NMS threshold, decoded output mode, `requires_grad` |
| `detr` | query class score, Hungarian loss, duplicate query selection |

## 13. Model 선택 기준

external model을 선택할 때는 다음 기준을 사용한다.

| 요구 | 적합한 시작점 |
| --- | --- |
| mature segmentation architecture 비교 | `torchseg` |
| torchvision detector 세대별 비교 | `torchdet` |
| 빠른 one-stage detector와 native NMS | `yolo` |
| anchor-free set prediction과 transformer query | `detr` |

whole-model은 architecture 내부를 세밀하게 바꾸기보다 검증된 결합을 유지하는 비교에 적합하다. backbone,
decoder, head를 독립적으로 ablation하려면 custom `seg` 또는 `det`가 더 직접적이다.

## 14. Code mapping

`torchseg` source는 다음과 같다.

| 역할 | 구현 |
| --- | --- |
| model과 classifier replacement | `src/models/torchseg/model.py` |
| mask target reuse | `src/models/torchseg/preprocessor.py` |
| mask corner decode reuse | `src/models/torchseg/postprocessor.py` |
| BCE, Dice, optimizer | `src/models/torchseg/wrapper.py` |

detector source는 package별로 같은 file 구성을 사용한다.

| model | model | target | decode | wrapper |
| --- | --- | --- | --- | --- |
| `torchdet` | `src/models/torchdet/model.py` | `preprocessor.py` | `postprocessor.py` | `wrapper.py` |
| `yolo` | `src/models/yolo/model.py` | `preprocessor.py` | `postprocessor.py` | `wrapper.py` |
| `detr` | `src/models/detr/model.py` | `preprocessor.py` | `postprocessor.py` | `wrapper.py` |

## 15. 실행 예시

torchvision segmentation은 다음과 같이 실행한다.

```bash
python scripts/train.py --model torchseg --network fcn_resnet50 --head mask --save
```

torchvision detection은 다음과 같이 실행한다.

```bash
python scripts/train.py --model torchdet --network fasterrcnn_resnet50_fpn --head point --save
```

YOLO와 DETR 예시는 다음과 같다.

```bash
python scripts/train.py --model yolo --network yolov8n --head box --save
python scripts/train.py --model detr --network detr_resnet50 --head point --save
```

evaluation과 prediction에서는 training과 동일한 assembly option과 checkpoint를 사용한다.

## 16. 핵심 요약

external whole-model은 mature architecture와 native loss를 유지하면서 project corner contract에 연결한다.
`torchseg`는 classifier를 1-channel mask로 바꾸고 custom segmentation target과 decode를 재사용한다.
`torchdet`, `yolo`, `detr`는 corner를 class별 pseudo-box로 바꾸지만 target dictionary, native loss,
candidate selection 방식은 서로 다르다. 이 model들을 비교할 때는 architecture 이름만 보지 말고 target
unit, label mapping, missing-class fallback, warmup과 native loss semantics를 함께 확인한다.
