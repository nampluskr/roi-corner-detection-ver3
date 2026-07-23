# Model Assembly

이 문서는 CLI의 `model`, `network`, `head`가 실제 Python object로 조립되는 과정을 설명한다. 초보자는
세 단어를 모두 model 이름으로 생각하기 쉽지만, project에서는 서로 다른 선택 축이다.

조립 규칙을 이해하면 어떤 option 조합이 가능한지, checkpoint를 load할 때 왜 같은 option을 다시
사용해야 하는지, external whole-model이 custom model과 어떻게 다른지 알 수 있다.

## 1. 조립의 네 단위

현재 project는 다음 단위를 구분한다.

| 단위 | 선택 질문 | 예시 |
| --- | --- | --- |
| model | corner를 어떤 표현과 loss로 학습할 것인가 | `reg`, `seg`, `peak` |
| network | 어떤 encoder 또는 complete architecture를 쓸 것인가 | `custom`, `resnet18`, `yolov8n` |
| head | model 안에서 어떤 output variant를 쓸 것인가 | `gap`, `mask`, `box` |
| wrapper | 학습과 추론을 어떤 lifecycle로 실행할 것인가 | `RegWrapper` |

CLI 사용자는 앞의 세 축을 선택한다. wrapper는 `model` 선택에 따라 factory가 결정한다.

## 2. `model`의 의미

`model`은 `src/models/<model>/` package와 factory dispatch key다. 단순 architecture 이름이 아니라 target,
raw output, loss, postprocess를 포함하는 task 표현을 선택한다.

예를 들어 `--model seg`는 다음 묶음을 선택한다.

```text
SegModel
SegPreprocessor
SegPostprocessor
SegWrapper
BCE and Dice losses
```

network를 ResNet에서 custom으로 바꿔도 mask target과 corner decode는 `seg` 계약을 유지한다.

## 3. `network`의 두 의미

`network`는 model 계열에 따라 의미가 다르다.

| 조립 계열 | `network` 의미 |
| --- | --- |
| composable model | feature를 추출하는 backbone 이름 |
| external whole-model | segmentation 또는 detection complete architecture 이름 |

`--model reg --network resnet18`에서 network는 coordinate head 앞의 encoder다. 반면
`--model yolo --network yolov8n`에서는 YOLO 전체 architecture와 local checkpoint를 가리킨다.

## 4. `head`의 의미

head는 model별 output variant다. 모든 model에 동일한 head 이름을 사용할 수 없다.

| model | head | 실제 의미 |
| --- | --- | --- |
| `reg` | `gap`, `spatial` | coordinate feature aggregation |
| `seg` | `mask` | single-channel ROI mask |
| `peak` | `peak` | four Gaussian point maps |
| `ridge` | `ridge` | four Gaussian edge maps |
| `det` | `box`, `point` | regression channel 구성 |
| `gcn` | `gcn` | iterative graph corner sequence |
| `hybrid` | `hybrid` | mask와 geometry pipeline |
| `torchseg` | `mask` | CLI compatibility value |
| external detector | `box`, `point` | pseudo-box target size |

external detector의 `head=point`는 YOLO나 DETR 내부 architecture를 point detector로 교체하지 않는다.
pseudo-box size를 더 작게 설정할 뿐 final prediction은 box center다.

## 5. CLI에서 factory까지

script 실행 시 조립 흐름은 다음과 같다.

```text
command line
-> scripts.config.parse_args
-> argparse Namespace
-> get_wrapper_kwargs
-> src.core.factory.get_wrapper(model, **kwargs)
-> model-specific Wrapper
-> Model + Preprocessor + Postprocessor
```

`parse_args`는 모든 script가 공유한다. `get_wrapper_kwargs`는 현재 `network`, `head`, `warmup_epochs`만
wrapper constructor로 전달한다. `get_wrapper`는 model 문자열을 `if` branch로 dispatch한다.

registry에 없는 model은 `NotImplementedError`를 발생시킨다. network와 head 검증은 주로 model 또는 wrapper
constructor에서 수행한다.

## 6. Model registry

현재 factory가 지원하는 11개 model은 다음과 같다.

| model | wrapper | raw output | 기본 head |
| --- | --- | --- | --- |
| `reg` | `RegWrapper` | `(B, 8)` logits | `gap` |
| `seg` | `SegWrapper` | binary mask logits | `mask` |
| `det` | `DetWrapper` | class and regression maps | `box` |
| `peak` | `PeakWrapper` | Gaussian peak logits | `peak` |
| `ridge` | `RidgeWrapper` | Gaussian ridge logits | `ridge` |
| `gcn` | `GCNWrapper` | corner refinement sequence | `gcn` |
| `hybrid` | `HybridWrapper` | binary mask logits | `hybrid` |
| `torchseg` | `TorchSegWrapper` | torchvision mask logits | `mask` |
| `torchdet` | `TorchDetWrapper` | torchvision detections | `box` |
| `yolo` | `YoloWrapper` | Ultralytics detections | `box` |
| `detr` | `DetrWrapper` | Hugging Face DETR output | `box` |

## 7. Composable model

composable model은 project component를 조립한다.

```text
backbone
-> adapter
-> FeatureBundle
-> optional decoder or neck
-> task head
-> raw output
```

`reg`, `seg`, `det`, `peak`, `ridge`, `gcn`, `hybrid`이 이 계열이다. 같은 backbone을 여러 model에서
재사용하고 decoder나 head만 바꿀 수 있다.

## 8. Backbone

backbone은 image를 feature로 바꾸는 encoder다. 현재 세 source가 있다.

| source | 예시 | weight 방식 |
| --- | --- | --- |
| project custom | `custom` | from scratch |
| torchvision | `resnet18`, `vit_b_16`, `swin_t` | local `.pth` |
| timm | registered Wide ResNet, DeiT, CaiT | local safetensors |

`CustomBackbone`은 stride `(2, 4, 8, 16)`의 네 stage를 만든다. pretrained backbone은 architecture마다 stage
channel과 stride가 다르며 wrapper class가 이를 common metadata로 노출한다.

## 9. Adapter와 `FeatureBundle`

CNN은 feature map을, transformer는 class token과 patch token을 반환한다. adapter는 이 native 차이를
`global_feature`, `spatial_feature`, `stages` field로 바꾼다.

consumer는 backbone class를 직접 검사하지 않고 필요한 feature field를 사용한다. 이 경계가 있기 때문에
`reg`의 gap head는 CNN global pooling과 transformer class token을 같은 의미의 input으로 받을 수 있다.

## 10. Capability 기반 조합

network 이름이 registry에 있다고 해서 모든 model과 결합되는 것은 아니다.

| 필요한 capability | consumer | 가능한 feature |
| --- | --- | --- |
| global | `reg` gap | pooled CNN final 또는 class token |
| spatial | `reg` spatial, `gcn` | CNN final map 또는 reshaped tokens |
| stages | `seg`, `peak`, `ridge`, `det`, `hybrid` | multi-resolution CNN stage list |

현재 dense decoder와 neck은 stage list가 필요하므로 ViT token만 제공하는 조합을 거부한다. `det` source는
ViT family에 대해 capability가 없다는 구체적인 오류를 제공한다.

## 11. Decoder, neck, head

backbone 이후 component의 책임은 다음과 같다.

| component | 역할 | 사용 model |
| --- | --- | --- |
| `UNetDecoder` | deep feature를 upsample하고 shallow skip을 더함 | `seg`, `peak`, `ridge`, `hybrid` |
| `MultiScaleNeck` | detection stride까지 top-down fusion | `det` |
| coordinate head | feature를 8 logits으로 projection | `reg` |
| dense head | decoded feature를 4 map으로 projection | `peak`, `ridge` |
| mask head | decoded feature를 1 map으로 projection | `seg`, `hybrid` |
| detection head | class와 regression branch 생성 | `det` |

`gcn`은 별도 decoder 대신 spatial feature에서 initial corner를 만들고 graph refiner를 반복한다.

## 12. External whole-model

`torchseg`, `torchdet`, `yolo`, `detr`은 library 내부 encoder-decoder-head 결합을 유지한다. project가
교체하거나 추가하는 부분은 다음과 같다.

```text
local pretrained whole model
-> task classifier replacement
-> project preprocessor for native labels
-> native training loss in wrapper
-> project postprocessor for final corners
```

이 계열은 `FeatureBundle`을 사용하지 않는다. wrapper가 native image list, label dictionary, output object를
common trainer step으로 감싼다.

## 13. 조립 예시: custom regression

다음 command를 보자.

```bash
python scripts/train.py --model reg --network custom --head gap
```

조립 결과는 다음과 같다.

```text
RegWrapper
-> RegModel
   -> CustomBackbone
   -> CNNBackboneAdapter(global only)
   -> GapHead
-> RegPreprocessor
-> RegPostprocessor
-> WingLoss
```

## 14. 조립 예시: pretrained segmentation

다음 command는 project decoder를 사용하면서 pretrained encoder를 선택한다.

```bash
python scripts/train.py --model seg --network resnet18 --head mask
```

조립 결과는 `TorchBackbone(resnet18)`, stage adapter, `UNetDecoder`, `MaskHead`, segmentation target과
postprocessor다. `torchseg`와 달리 torchvision FCN whole model을 사용하는 것은 아니다.

## 15. 조립 예시: external YOLO

다음 command는 complete YOLO model을 선택한다.

```bash
python scripts/train.py --model yolo --network yolov8n --head point
```

`YoloWrapper`는 local YOLOv8n checkpoint, 4-class classifier, point-size pseudo-box target, native loss와 NMS
postprocessor를 조립한다. project `MultiScaleNeck`과 `DetectionHead`는 사용하지 않는다.

## 16. Default 조합의 주의점

global parser default는 `model=reg`, `network=custom`, `head=gap`이다. 이 default는 `reg`에는 맞지만 다른
model에 자동으로 맞춰지지 않는다.

예를 들어 `--model peak`만 지정하면 head default `gap`이 전달되어 `PeakWrapper`가 오류를 발생시킨다.
`--model yolo`만 지정하면 network default `custom`이 전달되어 unknown YOLO network 오류가 발생한다.

따라서 model을 변경할 때는 항상 compatible `network`와 `head`를 함께 명시한다.

## 17. Image size 전달의 현재 한계

parser의 `--image_size`는 dataloader resize에는 사용되지만 `get_wrapper_kwargs`가 wrapper에 전달하지 않는다.
wrapper들은 constructor default 224를 유지한다. dense target size와 detection pixel conversion이 wrapper
image size에 의존하므로 현재 CLI에서 224 이외 값을 사용하는 것은 안전한 일반 기능이 아니다.

이 제한은 조립 layer의 current behavior이며 향후 code 변경 시 factory forwarding과 model별 shape test가
함께 필요하다.

## 18. Warmup 조립

`get_wrapper_kwargs`는 지원 model에 `warmup_epochs`를 전달한다. 다만 실제 적용 여부는 wrapper가
`applied_warmup_epochs`를 설정하는지에 달려 있다.

대부분의 composable model과 external detector는 backbone freeze phase를 적용한다. `torchseg`는 공통
argument를 받지만 current implementation에서 phase warmup을 적용하지 않고 whole model을 `1e-4`로
학습한다.

## 19. Checkpoint identity

checkpoint는 model state dictionary만 저장한다. CLI option과 class metadata를 checkpoint 안에 별도
manifest로 저장하지 않는다. load할 때 사용자가 같은 `model`, `network`, `head`로 동일한 architecture를
다시 조립해야 한다.

option이 다르면 missing key, unexpected key, shape mismatch가 발생하거나 의미가 다른 architecture에
잘못 load할 수 있다. output directory에 assembly 이름을 포함하는 이유가 여기에 있다.

## 20. Output path와 assembly

기본 output path는 다음 구조다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

`network_head`는 `<network>_<head>`, experiment name은
`<model>_bs<batch_size>_ep<max_epochs>_<network>_<head>`다. assembly와 training scale을 path에 남겨
checkpoint identity를 사람이 추적할 수 있게 한다.

## 21. 잘못된 조합의 대표 증상

주요 오류와 원인은 다음과 같다.

| 오류 | 원인 |
| --- | --- |
| unknown network | model registry에 없는 network |
| unknown head | model과 compatible하지 않은 head |
| no stages capability | dense model에 token-only backbone 사용 |
| local weight not found | registered external asset 부재 |
| checkpoint shape mismatch | training과 load assembly 불일치 |
| target and output shape mismatch | non-default image size 전달 불일치 |

## 22. Code mapping

assembly를 확인할 source는 다음과 같다.

| 책임 | 구현 |
| --- | --- |
| CLI default와 wrapper kwargs | `scripts/config.py` |
| model dispatch | `src/core/factory.py` |
| backbone registry | `src/components/backbones.py` |
| native feature adaptation | `src/components/adapters.py` |
| feature capability | `src/components/features.py` |
| decoder, neck, head | `src/components/decoders.py`, `necks.py`, `heads.py` |
| model-specific composition | `src/models/<model>/model.py` |
| lifecycle composition | `src/models/<model>/wrapper.py` |

## 23. 핵심 요약

`model`은 corner 표현과 training contract를, `network`는 encoder 또는 whole architecture를, `head`는
model-specific output variant를 선택한다. factory는 model string으로 wrapper를 만들고 wrapper는 model,
preprocessor, postprocessor, loss를 묶는다. composable model은 `FeatureBundle` capability를 기준으로
component를 조립하고 external model은 native whole-model interface를 wrapper에서 변환한다. 다른 model을
실행할 때는 compatible한 세 option을 항상 함께 명시한다.
