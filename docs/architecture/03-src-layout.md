# Source Layout

이 문서는 project source가 왜 여러 directory로 나뉘어 있는지, 각 directory가 어떤 책임을 가지는지,
새로운 model이나 공통 component를 추가할 때 어느 file을 변경해야 하는지 설명한다. 처음 source를 읽는
사용자는 class 이름보다 먼저 책임 경계를 이해하는 것이 좋다. 같은 기능처럼 보여도 data를 읽는 코드와
model target을 만드는 코드는 서로 다른 이유로 변경되기 때문이다.

## 1. Directory를 나누는 이유

작은 실험 코드는 dataset, network, loss, training loop를 한 file에 작성해도 실행할 수 있다. 그러나 model
종류가 늘어나면 같은 data split과 metric을 여러 번 복사하게 되고, 한 model을 수정하다가 다른 model의
실행 흐름까지 바꾸기 쉽다.

현재 project는 다음 질문에 따라 source를 분리한다.

| 질문 | 담당 위치 |
| --- | --- |
| image와 corner를 어떻게 읽고 변환하는가 | `src/data/` |
| 재사용 가능한 neural network 부품은 무엇인가 | `src/components/` |
| 한 model 표현의 target, raw output, loss, decode는 무엇인가 | `src/models/<model>/` |
| model 종류와 관계없이 epoch, 평가, 저장을 어떻게 실행하는가 | `src/core/` |
| geometry, image, file 처리 중 재사용할 계산은 무엇인가 | `src/utils/` |
| 사용자가 어떤 option으로 작업을 시작하는가 | `scripts/` |

이 분리는 file을 보기 좋게 정리하는 목적만 있지 않다. 상위 실행 계층은 model 내부 표현을 몰라도 되고,
model package는 CLI나 output directory를 몰라도 되는 의존 관계를 만든다.

## 2. Project의 큰 구조

source와 실행 진입점의 핵심 구조는 다음과 같다. folder는 먼저, file은 그 뒤에 표시했다.

```text
project-root/
├── scripts/
│   ├── batch_config.py
│   ├── batch_run.py
│   ├── config.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
└── src/
    ├── components/
    ├── core/
    ├── data/
    ├── models/
    ├── utils/
    └── __init__.py
```

`scripts/`는 사용자가 실행하는 얇은 진입점이고, `src/`는 실제 기능을 구현하는 import 가능한 package다.
script는 argument를 해석하고 필요한 object를 조립한 뒤 작업을 core object에 위임한다.

## 3. 의존 방향

권장 의존 방향은 다음과 같다.

```text
scripts
-> src.core
-> src.models and src.data
-> src.components and src.utils
```

화살표 왼쪽의 계층이 오른쪽 계층을 사용한다. 반대 방향 import를 만들면 낮은 수준의 component가 특정
script나 model을 알아야 하므로 재사용성이 사라진다.

현재 경계를 이해할 때 다음 원칙이 중요하다.

- `src/components/`는 특정 `src.models.<model>` package를 import하지 않는다.
- `src/models/<model>/`은 shared component와 utility를 사용할 수 있다.
- `src/core/`는 wrapper의 공통 step을 호출하며 model raw output을 직접 해석하지 않는다.
- `src/data/`는 `reg`, `seg`, `det` 같은 model 이름을 알지 않는다.
- `scripts/`는 training loop를 직접 구현하지 않고 core와 factory를 호출한다.

예를 들어 segmentation mask target 생성은 모든 dataset이 공통으로 해야 하는 일이 아니다. 따라서
`src/data/dataset.py`가 아니라 `src/models/seg/preprocessor.py`에 둔다. 반면 CSV의 corner 순서를 읽는 일은
모든 model이 공유하므로 `src/data/dataset.py`가 담당한다.

## 4. `src/components/`: 재사용 가능한 부품

`src/components/`는 여러 model이 함께 사용하는 neural network와 계산 부품을 둔다. 현재 file별 책임은
다음과 같다.

| file | 책임 | 대표 개념 |
| --- | --- | --- |
| `adapters.py` | backbone 고유 output을 공통 feature로 변환 | CNN map, transformer token |
| `backbones.py` | custom, torchvision, timm encoder 생성 | feature extraction |
| `blocks.py` | 작은 network building block | convolution block |
| `decoders.py` | stage feature를 높은 resolution으로 복원 | `UNetDecoder` |
| `features.py` | feature data와 capability 계약 | `FeatureBundle`, `FeatureSpec` |
| `heads.py` | feature를 task raw output으로 projection | coordinate, mask, dense map |
| `losses.py` | 여러 model이 사용하는 stateful loss | BCE, Dice, focal, coordinate loss |
| `metrics.py` | final corner 공통 평가 | IoU, MCD, PCK, success rate |
| `necks.py` | multi-scale feature 융합 | detection pyramid |

component는 특정 model의 전체 의미를 소유하지 않는다. `UNetDecoder`는 feature를 upsample하지만 결과가
mask인지 peak map인지 결정하지 않는다. 마지막 channel 수와 target 의미는 model package가 head와
preprocessor를 조합해 결정한다.

### 4.1 Backbone과 adapter의 분리

backbone마다 native output 형식이 다르다. CNN은 보통 `(B, C, H, W)` map을 만들고 transformer는
`(B, N, C)` token sequence를 만들 수 있다. adapter는 이를 `FeatureBundle`의 `global_feature`,
`spatial_feature`, `stages`로 변환한다.

이 분리 덕분에 consumer는 `isinstance(backbone, ResNet)` 같은 검사를 하지 않고 필요한 capability를
요청한다. stage pyramid가 필요한 decoder는 `stages`를, coordinate gap head는 `global_feature`를 사용한다.

### 4.2 Component에 둘 코드의 기준

다음 조건을 만족하면 shared component가 적절한 위치일 가능성이 높다.

1. 둘 이상의 model이 같은 계산을 같은 의미로 사용한다.
2. model-specific target이나 postprocess 규칙을 몰라도 동작한다.
3. input과 output tensor 계약을 독립적으로 설명할 수 있다.

반대로 이름만 비슷하고 target 의미나 failure 처리 방식이 다르면 각 model package에 두는 편이 명확하다.

## 5. `src/models/base/`: 공통 model 계약

base package는 모든 model이 반드시 같은 class hierarchy를 사용하도록 강제하기보다, 공통 역할과 기본
lifecycle을 제공한다.

```text
src/models/base/
├── __init__.py
├── model.py
├── postprocessor.py
├── preprocessor.py
└── wrapper.py
```

각 file의 의미는 다음과 같다.

| file | 입력 | 출력 또는 책임 |
| --- | --- | --- |
| `model.py` | image tensor | model-specific raw output |
| `preprocessor.py` | common corners | model-specific training target |
| `postprocessor.py` | raw output | common final corners |
| `wrapper.py` | model과 batch | optimizer, loss, metric, step lifecycle |

`BaseWrapper`는 device 이동, loss와 metric state reset, 일반적인 train, validation, prediction step을
구현한다. native library가 다른 호출 규약을 요구하는 external model은 필요한 step만 override한다.

## 6. `src/models/<model>/`: 표현 단위의 package

model package 하나는 architecture 이름 하나가 아니라 corner를 학습하는 표현 하나를 소유한다. 예를 들어
`seg` package는 polygon mask target, mask loss, mask에서 corner를 복원하는 규칙을 함께 가진다.

대부분의 package는 다음 구조를 따른다.

```text
src/models/<model>/
├── __init__.py
├── model.py
├── postprocessor.py
├── preprocessor.py
└── wrapper.py
```

일부 package에는 네 file이 모두 필요하지 않다. common corner target을 그대로 쓰면 base preprocessor를
재사용할 수 있고, wrapper 안에서 native model을 직접 다루면 별도 `model.py`가 없을 수도 있다. 중요한 것은
file 수를 맞추는 일이 아니라 역할의 경계를 보존하는 일이다.

### 6.1 `model.py`

`model.py`는 image에서 raw output을 계산한다. CSV, output path, argparse를 알지 않는다. composable model은
backbone, adapter, decoder 또는 neck, head를 연결한다. external whole-model은 library architecture를
project가 사용할 형태로 감싼다.

### 6.2 `preprocessor.py`

preprocessor는 labeled corner `(B, 4, 2)`를 loss가 요구하는 target으로 바꾼다. mask rasterization,
Gaussian peak, ridge, detection pseudo-box가 여기에 해당한다. inference에는 정답이 없으므로 이 단계는
사용되지 않는다.

### 6.3 `postprocessor.py`

postprocessor는 raw output을 `(B, 4, 2)` final corner로 바꾼다. sigmoid, argmax, threshold, contour,
line fitting, box center 추출 같은 규칙을 포함할 수 있다. core evaluator가 모든 model을 같은 metric으로
평가할 수 있는 이유가 이 경계에 있다.

### 6.4 `wrapper.py`

wrapper는 model, preprocessor, postprocessor, optimizer, scheduler, loss, metric을 하나의 실행 단위로
조립한다. wrapper가 반환하는 train 결과는 running loss와 metric dictionary이고, prediction 결과는 CPU
NumPy corner array다.

## 7. `src/data/`: model과 독립적인 data pipeline

data package의 구조는 다음과 같다.

```text
src/data/
├── __init__.py
├── dataloader.py
├── dataset.py
└── transforms.py
```

`dataset.py`는 CSV 행을 image path와 normalized corner로 해석한다. `dataloader.py`는 sample을 batch로
묶고 train split에 shuffle과 `drop_last`를 적용한다. `transforms.py`는 image와 corner를 함께 변환하는
joint transform을 제공한다.

여기서 target은 model-specific mask나 box가 아니라 항상 common corner다. model 표현은 data loading이
끝난 뒤 wrapper의 preprocessor가 만든다. 따라서 같은 split을 여러 model이 동일하게 사용할 수 있다.

## 8. `src/core/`: 실행 lifecycle

core package는 model 내부 수학보다 작업 단위의 반복과 저장을 담당한다.

| file | 책임 | model에 요구하는 interface |
| --- | --- | --- |
| `evaluator.py` | test split 공통 metric 계산과 `metrics.json` 저장 | `predict_step` |
| `factory.py` | transform, dataset, dataloader, wrapper, logger 생성 | model registry |
| `predictor.py` | sample별 target과 prediction 수집, CSV 저장 | `predict_step` |
| `trainer.py` | epoch 반복, validation, scheduler, early stopping, history 저장 | train/eval lifecycle |

`Trainer`는 `seg` mask logit이나 YOLO detection object를 해석하지 않는다. wrapper가 `train_step`과
`eval_step` 안에서 native 차이를 처리하기 때문이다. `Evaluator`도 raw output 대신 final corner만 받는다.

### 8.1 Factory의 역할

`src/core/factory.py`는 문자열 option을 object로 바꾸는 composition root다. `get_wrapper("seg", ...)`는
필요한 package를 lazy import하고 `SegWrapper`를 만든다. `get_dataloader("train", ...)`는 split에 맞는
transform과 sampling 옵션을 적용한다.

새 model을 registry에 추가하지 않으면 package를 작성해도 CLI에서 선택할 수 없다. 반대로 factory가 model
내부 component를 하나씩 조립하기 시작하면 package 책임이 분산된다. 현재는 wrapper constructor가 내부
조립을 소유한다.

## 9. `src/utils/`: domain helper

utility는 여러 계층에서 반복되는 작고 독립적인 기능을 둔다. 현재 주요 범주는 geometry, image, IO,
visualization이다.

utility에 코드를 둘 때는 호출 방향을 주의해야 한다. 예를 들어 `save_model`은 model 종류와 관계없이
`state_dict`를 저장하므로 utility가 적절하다. 특정 model의 threshold와 class mapping을 utility에 두면
model 계약이 package 밖으로 흩어지므로 적절하지 않다.

## 10. `scripts/`: 사용자 진입점

script는 다음 세 단계만 명확하게 수행한다.

```text
parse CLI
-> build dataloader and wrapper
-> delegate to Trainer, Evaluator, or Predictor
```

각 실행 file의 역할은 다음과 같다.

| file | 역할 |
| --- | --- |
| `batch_config.py` | batch experiment dictionary 목록 |
| `batch_run.py` | config별 subprocess 실행 |
| `config.py` | 공통 default, argument parser, output identity |
| `evaluate.py` | checkpoint를 load하고 test metric 저장 |
| `predict.py` | checkpoint를 load하고 sample prediction 저장 |
| `train.py` | train과 valid loader로 학습하고 선택적으로 checkpoint 저장 |

script는 project root를 `sys.path`에 추가한 뒤 `scripts.xxx`, `src.xxx` absolute import를 사용한다. 이 규칙은
어느 working directory에서 import가 시작되는지 명확하게 한다.

## 11. 한 batch가 source를 통과하는 예

`reg` training batch를 예로 들면 source 책임은 다음 순서로 연결된다.

```text
scripts/train.py
-> src.core.factory.get_dataloader
-> src.data.dataset.CornerDataset
-> src.data.transforms.Compose
-> src.data.dataloader.Dataloader
-> src.core.factory.get_wrapper
-> src.models.reg.wrapper.RegWrapper
-> src.models.reg.model.RegModel
-> src.components backbone, adapter, head
-> src.core.trainer.Trainer
```

dataset은 `(image, corners)` sample을 만들고 dataloader는 `(B, 3, H, W)`와 `(B, 4, 2)` batch를 만든다.
trainer는 wrapper의 step을 호출한다. wrapper는 raw output과 target으로 loss를 계산하고 postprocessor로
final corner를 만든다. 각 계층은 바로 다음 경계에 필요한 정보만 전달한다.

## 12. 새 model을 추가하는 절차

새로운 표현을 추가할 때는 다음 순서를 권장한다.

1. common input과 final output이 model contract를 만족하는지 정의한다.
2. `src/models/<model>/` package와 빈 `__init__.py`를 만든다.
3. corner에서 model-specific target을 만드는 preprocessor를 구현한다.
4. image에서 raw output을 만드는 model을 구현한다.
5. raw output에서 ordered normalized corner를 만드는 postprocessor를 구현한다.
6. loss, metric, optimizer, scheduler를 조립하는 wrapper를 구현한다.
7. 재사용 가치가 분명한 block만 `src/components/`로 이동한다.
8. `src/core/factory.py`의 model registry에 wrapper를 추가한다.
9. `scripts/batch_config.py`에 실행 가능한 조합 예시를 추가한다.
10. target shape, raw output shape, final corner와 train/eval/predict smoke test를 확인한다.

model별 자세한 표현은 [model 문서](../models/README.md)를 참고하고, object 조립 과정은
[Model Assembly](02-model-assembly.md)를 참고한다.

## 13. 기존 model에 network를 추가하는 절차

새 model 표현이 아니라 새 backbone만 추가한다면 model package를 복사할 필요가 없다. 일반적으로 다음
경계를 확인한다.

1. `src/components/backbones.py`가 architecture와 local weight를 만들 수 있어야 한다.
2. `src/components/adapters.py`가 native output을 common feature로 변환해야 한다.
3. `FeatureSpec`이 channel, stride, stage capability를 정확히 기록해야 한다.
4. 사용하려는 head, decoder, neck이 요구하는 capability를 제공해야 한다.
5. wrapper가 해당 network 이름을 허용하고 필요한 warmup optimizer group을 구성해야 한다.

network가 registry에 존재한다는 사실만으로 모든 model과 호환되지는 않는다. 예를 들어 stage pyramid가
없는 adapter는 `reg`에는 쓸 수 있어도 U-Net decoder 계열에는 쓸 수 없다.

## 14. 흔한 경계 위반

대표적인 경계 위반과 결과는 다음과 같다.

| 경계 위반 | 나타나는 문제 | 적절한 위치 |
| --- | --- | --- |
| dataset에서 segmentation mask를 생성 | 다른 model도 불필요한 표현에 의존 | `models/seg/preprocessor.py` |
| model에서 output path를 생성 | 동일 model을 다른 실행 context에서 재사용하기 어려움 | `scripts/config.py` |
| trainer가 raw output type별로 분기 | model 추가마다 core 수정 범위가 커짐 | model wrapper |
| component가 특정 model package를 import | 순환 의존과 재사용 저하 | 의존 방향 재설계 |
| postprocessor가 metric state를 갱신 | decode와 평가가 결합 | wrapper 또는 evaluator |
| factory가 loss 세부값까지 조립 | model 의미가 여러 file에 분산 | model wrapper |

## 15. Source 탐색 순서

처음 특정 실행을 추적할 때는 다음 순서가 효율적이다.

1. `scripts/config.py`에서 option과 default를 확인한다.
2. `scripts/train.py`, `evaluate.py`, `predict.py` 중 실행 진입점을 확인한다.
3. `src/core/factory.py`에서 dataset과 wrapper 생성 경로를 확인한다.
4. 해당 `src/models/<model>/wrapper.py`에서 loss와 lifecycle을 확인한다.
5. 같은 package의 model, preprocessor, postprocessor를 확인한다.
6. wrapper가 import하는 shared component까지 내려간다.
7. 결과 저장 방식은 trainer, evaluator, predictor에서 확인한다.

이 순서를 사용하면 전체 source를 한 번에 읽지 않고도 CLI 입력이 어느 object와 tensor로 바뀌는지 추적할
수 있다.

## 16. Code mapping

구조 원칙을 확인할 핵심 source는 다음과 같다.

| 주제 | source |
| --- | --- |
| component와 feature 계약 | `src/components/` |
| 공통 model lifecycle | `src/models/base/` |
| model별 표현 | `src/models/<model>/` |
| CSV와 split | `src/data/dataset.py` |
| joint transform | `src/data/transforms.py` |
| dataloader 정책 | `src/data/dataloader.py` |
| object 조립 | `src/core/factory.py` |
| epoch 실행 | `src/core/trainer.py` |
| CLI와 output identity | `scripts/config.py` |

## 17. 핵심 요약

`src/data`는 common sample, `src.components`는 reusable building block, `src.models`는 model-specific 표현,
`src.core`는 실행 lifecycle, `src.utils`는 독립 helper를 소유한다. `scripts`는 이들을 조립해 사용자가
실행할 command를 제공한다. 새 기능을 추가할 때는 코드가 무엇을 계산하는지만 보지 말고, 어떤 계층이 그
의미를 소유해야 하는지 판단해야 한다.
