# src/models $\to$ src/methods 재구성 (method별 assembly 분리 + RegModel 통합)

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-21 |
| 완료일 | 2026-07-21 |
| 적용 범위 | ver3 `src/methods/` 신규, `src/components/decoders.py` 신규, `src/core/factory.py`·`scripts/config.py` 갱신 |
| 관련 문서 | [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0004-components-flatten-plan.md](0004-components-flatten-plan.md) |
| 선행 | 0004 (components flatten) 완료 |

## 1. 목적과 배경 (Context)

ver2의 모델 계층(`src/models/`)은 아직 ver3로 이관되지 않았다(0003 §4의 "미해결 의존"). 이 계층은
task(reg/seg/det/heatmap)별 폴더 안에 여러 assembly 변형(composable backbone 모델과 외부 whole-model)이
섞여 있다. 예를 들어 `seg/model.py`에 `SegModel`(backbone 조립형)과 `TorchSegModel`(torchvision 외부 모델)이,
`det/model.py`에 `DetModel/TorchDetModel/YoloDetModel/DetrDetModel` 4종이 한 파일에 공존한다. CLI는
`--method det --model yolov8n`처럼 `model` 문자열을 sniffing 해 변형을 고른다(factory `get_wrapper`).

ver3에서는 assembly 변형 자체를 method로 승격해 `src/methods/<method>/`로 1:1 분리한다. 각 method
폴더는 `model.py`, `wrapper.py`, `preprocessor.py`, `postprocessor.py`로 구성되어 자기 완결적이며,
factory는 `model` sniffing 없이 `method` 문자열만으로 dispatch 한다. 조립 재료는 0004에서 확립한
`src/components/*`(SSOT)를 import 한다.

동시에 두 가지 동작을 개선한다.

- RegModel 통합: reg를 `CustomRegModel`과 `TorchRegModel` 2종에서 `SegModel`/`DetModel`과 동일하게
  backbone 인자로 custom/timm/torchvision을 모두 다루는 단일 `RegModel`로 통합한다.
- custom backbone warmup 허용: reg/seg/det/heatmap에서 custom backbone일 때 warmup을 0으로 강제하던
  특수케이스를 제거한다. 이후 synthetic/measured 데이터 학습 시 이전에 학습된(custom 포함) 가중치를 불러와
  초기에 backbone을 freezing 할 수 있어야 하기 때문이다.

ver2 내부 파일은 참고와 읽기 전용이므로([../CLAUDE.md](../CLAUDE.md)) 결과는 ver3에만 생성하며 ver2는 변경하지 않는다.

## 2. 최종 method 구성

`src/methods/` 아래 9개 폴더(base와 8 method)를 둔다. 각 폴더는 빈 `__init__.py`와 아래 파일을 갖는다.

| method | 폴더 파일 | 주요 클래스 | ver2 출처 |
| --- | --- | --- | --- |
| base | base_model/base_wrapper/base_preprocessor/base_postprocessor | `BaseModel`,`BaseWrapper`,`BasePreprocessor`,`BasePostprocessor` | `models/base/*` |
| reg | model/wrapper/preprocessor/postprocessor | `RegModel`(통합),`RegWrapper`,`RegPreprocessor`,`RegPostprocessor` | `models/reg/*` |
| seg | 〃 | `SegModel`,`SegWrapper`,`SegPreprocessor`,`SegPostprocessor` | `models/seg/*` (SegModel 부분) |
| det | 〃 | `DetModel`,`DetWrapper`,`DetPreprocessor`,`DetPostprocessor` | `models/det/*` (DetModel 부분) |
| heatmap | 〃 | `HeatmapModel`,`HeatmapWrapper`,`HeatmapPreprocessor`,`HeatmapPostprocessor` | `models/heatmap/*` (ver2 방식 그대로) |
| torchseg | 〃 | `TorchSegModel`,`TorchSegWrapper`, seg mask pre/post 재사용 | `models/seg/*` (TorchSegModel 부분) |
| torchdet | 〃 | `TorchDetModel`,`TorchDetWrapper`,`TorchDetPreprocessor`,`TorchDetPostprocessor` | `models/det/*` (TorchDet 부분) |
| yolo | 〃 | `YoloModel`,`YoloWrapper`,`YoloPreprocessor`,`YoloPostprocessor` | `models/det/*` (YoloDet 계열, 이름에서 "Det" 제거) |
| detr | 〃 | `DetrModel`,`DetrWrapper`,`DetrPreprocessor`,`DetrPostprocessor` | `models/det/*` (DetrDet 계열, 이름에서 "Det" 제거) |

또한 `src/components/decoders.py`를 신규로 만들어 ver2의 segmentation decoder 구현을 `UNetDecoder`로 이동한다.
seg와 heatmap model이 공유하므로 조립 재료(components SSOT)로 승격한다.

## 3. RegModel 통합 상세

`methods/reg/model.py`는 ver2의 `_build_extractor_and_head` 헬퍼를 유지하고, 단일 `RegModel(BaseModel)`로
통합한다. `SegModel`의 backbone 분기 패턴을 그대로 따른다.

```python
class RegModel(BaseModel):
    def __init__(self, in_channels=3, backbone="custom", dropout=0.2, head="gap"):
        backbone = backbone or "custom"
        head = head or "gap"
        if backbone == "custom":
            encoder, is_vit = CustomBackbone(in_channels=in_channels), False
        elif backbone in SUPPORTED_BACKBONES:
            encoder, is_vit = TorchBackbone(backbone), backbone in VIT_BACKBONES
        elif backbone in SUPPORTED_TIMM_BACKBONES:
            encoder, is_vit = TimmBackbone(backbone), backbone in TIMM_VIT_BACKBONES
        else:
            raise ValueError(...)   # ("custom",) + SUPPORTED_BACKBONES + SUPPORTED_TIMM_BACKBONES
        self.head_name = head
        self.extractor, self.head = _build_extractor_and_head(encoder, backbone, is_vit, head, dropout)
    # forward(): ver2와 동일 (gap -> global_feature, spatial -> spatial_feature)
```

`CustomRegModel`과 `TorchRegModel`은 없앤다. `RegWrapper.build_model`도 `RegModel(...)` 단일 생성으로 대체한다.

## 4. custom backbone warmup 허용

`BaseWrapper`의 warmup 메커니즘(`on_fit_start`/`on_epoch_start`, `get_backbone_module()`이 `model.extractor`를
반환, `set_backbone_trainable`)은 그대로 둔다. reg/seg/det/heatmap wrapper에서 아래를 변경한다.

- `applied_warmup_epochs = 0 if custom ... else warmup_epochs` 특수케이스를 제거하고 항상
  `applied_warmup_epochs = warmup_epochs`로 둔다. custom도 `.extractor`를 가지므로 freeze/unfreeze가 정상 동작한다.
- `build_optimizer(phase)`는 `SegWrapper.build_optimizer` 패턴으로 통일한다. `warmup_epochs>0`이면
  phase1은 non-backbone만, phase2는 extractor(1e-5)와 non-backbone(1e-4)으로 구성한다. `warmup_epochs==0`이면
  `AdamW(model.parameters(), lr=1e-4)`로 custom from-scratch 기존 동작을 보존한다.
- det: ver2 `DetWrapper`는 warmup을 쓰지 않고 2-group 고정이었다. 이번에 seg 패턴과 동일하게
  `build_optimizer(phase)`와 `applied_warmup_epochs=warmup_epochs`를 도입해 custom warmup을 지원한다.
- torchseg/torchdet/yolo/detr는 ver2 warmup 동작을 그대로 유지한다(외부 whole-model, custom backbone 개념 없음).
  torchdet/yolo/detr는 이미 `get_backbone_module`/`get_backbone_layers` override로 warmup을 지원하고, torchseg는
  ver2대로 warmup을 비활성(flat AdamW)한다.

## 5. import 재작성 규칙 (모든 method 파일 공통)

0004의 components 경로와 신규 methods/decoders 경로로 일괄 치환한다.

| ver2 경로 | ver3 경로 |
| --- | --- |
| `src.models.base.base_*` | `src.methods.base.base_*` |
| `src.models.backbones.*` | `src.components.backbones` |
| `src.models.adapters.*` | `src.components.adapters` |
| `src.models.features` | `src.components.features` |
| `src.models.necks.multi_scale_neck` | `src.components.necks` |
| `src.models.heads.*` | `src.components.heads` |
| `src.models.blocks.*` | `src.components.blocks` |
| ver2 segmentation decoder module | `src.components.decoders` |
| `src.losses.*` | `src.components.losses` |
| `src.metrics.*` | `src.components.metrics` |
| `src.models.<method>.<x>` | `src.methods.<method>.<x>` |

method별 상수는 다음과 같이 분할한다. `SUPPORTED_DET_BACKBONES`는 `methods/det/model.py`,
`TORCHDET_*`는 `methods/torchdet/model.py`, `YOLODET_*`는 `methods/yolo/model.py`,
`DETRDET_*`는 `methods/detr/model.py`, `TORCHSEG_*`는 `methods/torchseg/model.py`로 옮긴다.
det의 pre/post에 있는 로컬 `NUM_CORNER_CLASSES`와 `BOX_CHANNELS`는 각 method 파일에 그대로 둔다.

torchseg는 seg와 동일한 mask target/추출을 쓴다. `methods/torchseg/preprocessor.py`와 `postprocessor.py`는
`TorchSegPreprocessor(SegPreprocessor)`, `TorchSegPostprocessor(SegPostprocessor)`처럼 빈 subclass로 정의해
(`from src.methods.seg.preprocessor import SegPreprocessor` 등) 4파일 구조를 유지하되 중복 로직을 피한다.
det 계열의 pre/post는 변형별로 서로 다르므로 각자 구현을 유지한다.

## 6. factory / CLI 갱신

[../../src/core/factory.py](../../src/core/factory.py)의 `get_wrapper`는 `model` sniffing을 제거하고
`method` 문자열만으로 dispatch 한다.

```text
reg -> RegWrapper, seg -> SegWrapper, det -> DetWrapper, heatmap -> HeatmapWrapper,
torchseg -> TorchSegWrapper, torchdet -> TorchDetWrapper, yolo -> YoloWrapper, detr -> DetrWrapper
```

각 분기는 `from src.methods.<method>.wrapper import <Wrapper>` 지연 import를 쓴다.

[../../scripts/config.py](../../scripts/config.py)의 `get_wrapper_kwargs`는 warmup 게이트 튜플을 8개 method
전체(`reg,seg,det,heatmap,torchseg,torchdet,yolo,detr`)로 확장한다. 모든 wrapper가 `warmup_epochs`를 받는다.
CLI 사용법이 바뀌어 외부 모델은 이제 `--method torchseg --model fcn_resnet50`, `--method yolo --model yolov8n`,
`--method torchdet --model fasterrcnn_resnet50_fpn`, `--method detr --model detr_resnet50` 형태로 지정한다.
exp/dir 이름 로직(`get_experiment`/`get_model_name`)은 `model or backbone` 기반이라 그대로 동작한다.
`scripts/batch_config.py`가 구 `(method=seg, model=fcn_resnet50)` 매핑을 열거하면 새 taxonomy로 갱신한다(구현 시 확인).

## 7. 코드 작성 규칙과 Python 경로 (ver2 지침 반영)

ver2 지침 파일(`../../../260712_roi-corner-detection-ver2/CLAUDE.md` §4 코드 작성 규칙, §5 실행 환경)의
내용은 ver3 [../../CLAUDE.md](../../CLAUDE.md) §3, §4, §5에 이미 반영되어 있으며, 이 plan의 모든 신규 Python
파일은 아래를 따른다.

- 식별자, 주석, docstring, 문자열에 한국어를 사용하지 않는다.
- 모든 파일 첫 줄은 `# src/methods/<method>/<file>.py: one-line description` 형식으로 쓰고, 그 다음 빈 줄
  하나를 둔 뒤 import를 작성한다.
- class와 top-level function은 한 줄 docstring을 쓰고, method에는 docstring을 쓰지 않는다.
- type hint를 쓰지 않고, 경로 처리는 `os.path`를 쓰며, 세로 정렬용 공백과 불필요한 주석을 넣지 않는다.
- `src/methods/`와 모든 하위 폴더에 빈 `__init__.py`를 둔다.
- `src/` 내부 import는 `src.xxx` absolute import를 쓴다(`src.components.*`, `src.methods.*`,
  `src.methods.base.*`).

Python 실행과 검증은 conda `pytorch_env`에서 ver3 project root
(`/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3`)를 기준으로 수행한다. `python -c`
같이 script의 `sys.path` 보정이 없는 명령은 `PYTHONPATH`에 project root를 포함한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
PYTHONPATH=/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3 python -c "import src"
```

## 8. 완료 기준

- `src/methods/`에 base와 8 method 폴더가 있고, 각 폴더에 빈 `__init__.py`와 4파일이 존재한다.
- `RegModel` 단일 클래스가 custom/timm/torchvision을 모두 처리하고 `CustomRegModel`/`TorchRegModel`이 없다.
- reg/seg/det/heatmap이 custom backbone에서도 `warmup_epochs>0`일 때 freeze 후 unfreeze로 동작한다.
- `src/components/decoders.py`에 `UNetDecoder`가 있고 seg와 heatmap이 이를 import 한다.
- factory가 `method`만으로 8개 wrapper를 dispatch 하고 `src.models.*` 참조 잔존이 없다.
- ver2 파일은 하나도 변경되지 않는다.

## 9. 검증 (conda `pytorch_env`)

컴파일은 `python -m py_compile src/methods/**/*.py src/components/decoders.py src/core/factory.py scripts/config.py`로 한다.

외부 가중치 없이 custom backbone으로 import와 구성을 확인한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
python -c "from src.core.factory import get_wrapper; \
  [get_wrapper(m, backbone='custom') for m in ('reg','seg','det','heatmap')]; print('OK composable')"
python -c "import torch; from src.core.factory import get_wrapper; \
  w=get_wrapper('reg', backbone='custom', warmup_epochs=1); w.on_fit_start(5); \
  print('reg custom warmup frozen:', not next(w.model.extractor.parameters()).requires_grad)"
```

두 번째 명령으로 custom backbone warmup(freeze) 동작을 직접 확인한다. 외부 모델은 로컬 가중치와
timm/ultralytics/transformers가 있을 때만 확인한다.

```bash
python -c "from src.core.factory import get_wrapper; get_wrapper('yolo', model='yolov8n'); print('OK yolo')"
```

미설치이거나 가중치가 없는 환경에서는 이 확인을 건너뛴다. 마지막으로 잔여 참조가 없는지
`grep -rn --include=*.py -E "src\.models\.|src\.(losses|metrics)\b" src/ scripts/`가 비어 있어야 한다.

## 10. 실행 결과 (2026-07-21)

conda `pytorch_env`에서 아래 항목을 확인했다.

- `py_compile`: base와 8 method의 전체 파일, `src/components/decoders.py`, `src/core/factory.py`,
  `scripts/config.py`, `scripts/batch_config.py` 모두 통과했다.
- composable 구성: `get_wrapper(m, backbone='custom')`이 reg, seg, det, heatmap에서 정상 생성됐다.
- custom warmup: reg, seg, det, heatmap 모두 `warmup_epochs=1`에서 `on_fit_start` 후
  `model.extractor`의 `requires_grad`가 False로 바뀌어 freeze 동작을 확인했다.
- 외부 method: torchseg, torchdet, yolo, detr의 model과 wrapper 모듈 import가 통과했다(가중치 필요
  구성은 로컬 가중치가 있을 때만 별도 확인한다).
- 잔여 참조: `grep -rn --include=*.py -E "src\.models\.|src\.(losses|metrics)\b" src/ scripts/`
  결과가 비어 있음을 확인했다.
- ver2 파일은 변경하지 않았다.
