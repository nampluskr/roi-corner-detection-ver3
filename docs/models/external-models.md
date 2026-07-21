# External Whole Models

`torchseg`, `torchdet`, `yolo`, `detr`는 project component를 다시 조립하지 않고, 외부 library의 whole
model을 네 corner task에 맞춘다. 이 계열은 native loss와 output 형식을 wrapper에서 처리한다.

## Registry

지원하는 external model은 다음과 같다.

| model | default network | native interface | target 표현 |
| --- | --- | --- | --- |
| `torchseg` | `fcn_resnet50` | torchvision segmentation | binary mask |
| `torchdet` | `fasterrcnn_resnet50_fpn` | torchvision detection | class pseudo-box |
| `yolo` | `yolov8n` | Ultralytics YOLO | class pseudo-box |
| `detr` | `detr_resnet50` | Hugging Face DETR | class pseudo-box |

각 model은 local pretrained weight 또는 snapshot 경로를 사용한다. network 이름이 registry에 없거나 local
asset이 없으면 wrapper 생성 시 오류가 발생한다.

## Segmentation

`torchseg`는 `SegPreprocessor`와 `SegPostprocessor`를 재사용한다. model output에서 mask logit을 선택하고,
`BCELoss`와 `DiceLoss`로 학습한다. `--head mask`만 지원한다.

## Detection

`torchdet`, `yolo`, `detr`는 네 corner를 class별 fixed-size pseudo-box로 변환한다. postprocessor는 각
class에서 score가 가장 높은 detection 또는 query를 선택하고 box center를 normalized corner로 바꾼다.

`torchdet`는 torchvision native detection loss, `yolo`는 Ultralytics detection loss, `detr`는 Hungarian
matching loss를 사용한다. `box`와 `point` head는 pseudo-box size를 각각 큰 값 또는 작은 값으로 선택하는
wrapper option이다.

## Operational Notes

external model은 custom model보다 dependency와 local weight 상태에 민감하다. checkpoint를 학습한 뒤에는
evaluation과 prediction에서 같은 `--model`, `--network`, `--head`를 사용해야 state와 native output
contract가 유지된다.
