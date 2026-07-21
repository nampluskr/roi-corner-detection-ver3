# Model Assembly

이 문서는 `roi-corner-detection-ver3`에서 model을 선택하고 조립하는 canonical 기준이다. 모든 실행은
`--model`, `--network` 또는 `--net`, `--head`의 세 축으로 표현하며, `src.core.factory.get_wrapper`가
선택된 model package의 wrapper를 생성한다.

## 조립 단위

각 용어는 서로 다른 책임을 가진다.

| 단위 | 책임 | 예시 |
| --- | --- | --- |
| model | corner 표현과 학습 규약을 선택한다 | `reg`, `seg`, `det` |
| network | encoder 또는 external whole-model architecture를 선택한다 | `custom`, `resnet18`, `yolov8n` |
| head | model 내부의 출력 표현 세부 구성을 선택한다 | `gap`, `mask`, `box` |
| wrapper | 학습, 평가, 예측의 실행 규약을 제공한다 | `SegWrapper` |

`model`은 디렉터리와 factory dispatch의 기본 단위다. `network`는 composable model에서는 backbone을,
external whole-model에서는 로컬 pretrained model 이름을 의미한다. `head`는 모든 model에 동일한 의미를
강제하지 않으며, 해당 model이 지원하는 raw output 변형만 선택한다.

## 공통 조립 흐름

일반적인 composable model의 계산 흐름은 다음과 같다.

```text
images -> backbone -> adapter -> FeatureBundle -> decoder or neck -> head -> raw output
corners -> preprocessor -> method target
raw output -> postprocessor -> normalized corners
```

`BaseWrapper`는 raw output과 method target을 loss에 전달하고, postprocessor가 복원한 normalized corner를
metric에 전달한다. 따라서 model은 task-specific raw output을 만들고, target 생성 및 최종 corner 복원은
각각 preprocessor와 postprocessor가 맡는다.

## Model Registry

현재 factory가 지원하는 model은 다음과 같다.

| model | raw output 표현 | 조립 방식 | 기본 head |
| --- | --- | --- | --- |
| `reg` | 8개 coordinate logit | composable | `gap` |
| `seg` | binary mask logit | composable | `mask` |
| `det` | class map과 box or point map | composable | `box` |
| `peak` | 4-channel Gaussian peak map | composable | `peak` |
| `ridge` | 4-channel Gaussian ridge map | composable | `ridge` |
| `gcn` | 초기값과 반복 정제 corner | composable refinement | `gcn` |
| `hybrid` | binary mask logit | composable with geometry | `hybrid` |
| `torchseg` | torchvision segmentation output | external whole-model | `mask` |
| `torchdet` | torchvision detection output | external whole-model | `box` |
| `yolo` | Ultralytics detection output | external whole-model | `box` |
| `detr` | Hugging Face DETR output | external whole-model | `box` |

`reg`, `seg`, `det`, `peak`, `ridge`, `gcn`, `hybrid`은 `src.components`의 backbone, adapter, decoder,
neck, head를 조합한다. `torchseg`, `torchdet`, `yolo`, `detr`은 외부 model의 내부 구조를 유지하고,
wrapper와 preprocessor, postprocessor로 project 계약에 맞춘다.

## Network와 Head

composable model의 `network`는 `custom`, torchvision backbone, 또는 timm backbone 이름을 받을 수 있다.
실제 지원 여부는 model이 요구하는 feature capability와 설치된 dependency에 따라 결정된다. dense model은
stage feature가 필요하고, coordinate regression은 global 또는 spatial feature를 사용한다.

`head`의 현재 선택 범위는 다음과 같다.

| model | head | 의미 |
| --- | --- | --- |
| `reg` | `gap`, `spatial` | global pooling 또는 spatial pooling coordinate head |
| `seg`, `torchseg` | `mask` | binary ROI mask |
| `det`, `torchdet`, `yolo`, `detr` | `box`, `point` | pseudo-box 크기를 포함하거나 점 크기로 학습하는 detection target |
| `peak` | `peak` | corner별 Gaussian peak target |
| `ridge` | `ridge` | corner별 Gaussian ridge target |
| `gcn` | `gcn` | graph refinement output |
| `hybrid` | `hybrid` | mask와 geometry refinement 조합 |

external whole-model의 `head`는 target box 크기를 결정하는 wrapper option이다. external network의 native
classifier 또는 detection head 구조를 CLI로 교체하는 기능은 제공하지 않는다.
