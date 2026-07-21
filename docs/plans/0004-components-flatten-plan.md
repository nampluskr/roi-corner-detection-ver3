# losses/metrics/조합용 모듈 → src/components/ flatten

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-21 |
| 적용 범위 | ver3 `src/components/` 신규, `src/core/evaluator.py` import 갱신 |
| 관련 문서 | [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0003-src-core-data-utils-plan.md](0003-src-core-data-utils-plan.md) |

## 1. 목적과 배경

ver2의 재사용 구성요소는 여러 하위 폴더에 파일 단위로 흩어져 있다. `src/losses/`(7개 파일),
`src/metrics/`(4개 파일), 그리고 `src/models/` 아래 모델을 "조합"하는 building-block 모듈들
(`heads/`, `backbones/`, `adapters/`, `necks/`, `blocks/`, `features.py`)이다. 이 폴더 계층은
빈 `__init__.py`에 전부 full-path import(`from src.models.blocks.conv_block import ConvBlock`)로만
연결되어 있어, 파일 수에 비해 얻는 격리 이점이 적다.

ver3에서는 이 재사용 구성요소를 도메인별 단일 모듈로 flatten 하여 `src/components/` 아래에 모은다.
이렇게 하면 `src.components.losses`, `src.components.backbones`처럼 도메인 단위로 import 되고,
후속 모델 재구성 plan에서 조립 대상(reg/seg/heatmap/det, `models/base/`)과 조립 재료(components)가
경로상 명확히 분리된다.

주 재구성 대상인 `XXXModel`/`XXXWrapper`(`src/models/reg|seg|heatmap|det`, `models/base/`)는 이
plan의 범위가 아니며 후속 plan에서 다룬다. 이 plan은 그 모델들이 조립에 쓰는 공용 재료만 확립한다.

ver2 내부 파일은 참고·읽기 전용이므로([../CLAUDE.md](../CLAUDE.md)) 결과는 ver3 내부에만 생성했으며,
ver2에는 어떤 변경도 가하지 않았다.

## 2. 범위

포함:
- `src/components/__init__.py` 신규(빈 파일, ver2의 빈 `__init__` 관례 유지)
- 아래 8개 flatten 모듈 신규 생성
- `src/core/evaluator.py`의 metrics import 3줄 갱신

제외 (후속 plan — 주 재구성 대상):
- `src/models/reg|seg|heatmap|det/` (`XXXModel`/`XXXWrapper`)
- `src/models/base/` (base_model/base_wrapper/base_preprocessor/base_postprocessor)
- `__pycache__` 등 컴파일 산출물은 이관하지 않는다

## 3. flatten 매핑

각 대상 모듈은 여러 ver2 파일을 의존 순서대로 이어붙여 만든다. 같은 파일 안으로 병합된 클래스에
대한 intra import는 제거하고, 다른 component를 가리키는 cross import는 새 경로로 재작성한다(§4).

| ver3 신규 (`src/components/`) | ver2 원본 (`src/`) | 병합 순서(선행 → 후행) |
| --- | --- | --- |
| `losses.py` | `losses/*.py` | base_loss → bce/dice/focal/heatmap_mse/smoothl1/wing |
| `metrics.py` | `metrics/*.py` | base_metric → corner_distance → polygon_iou → success_rate |
| `blocks.py` | `models/blocks/*.py` | conv_block → deconv_block |
| `features.py` | `models/features.py` | (단일 파일, 이동만) |
| `backbones.py` | `models/backbones/*.py` | base_backbone → custom/timm/torch |
| `adapters.py` | `models/adapters/*.py` | base_adapter → cnn/transformer |
| `necks.py` | `models/necks/*.py` | multi_scale_neck (단일) |
| `heads.py` | `models/heads/*.py` | coordinate/detection/heatmap/mask |

병합 시 유지 사항:
- 모든 클래스/함수/모듈 상수(예: `backbones.py`의 `SUPPORTED_BACKBONES`, `RESNET_BACKBONES`,
  `TIMM_CNN_BACKBONES` 등, `heads.py`의 `NUM_CORNER_CLASSES`, `BOX_CHANNELS`)를 그대로 보존한다.
- 각 파일 최상단 헤더 주석은 병합 파일의 새 헤더 1줄로 대체하고, 원본 파일별 설명은 섹션 구분
  주석(`# --- ... ---`)으로 유지한다.
- `torch`, `torch.nn` 등 외부 라이브러리 import는 병합 파일 상단에 중복 없이 모은다.
- `metrics.py`는 `from src.utils.geometry import polygon_area`를 유지한다(`utils`는 이동 대상 아님).

이관 후 구조:

```text
src/components/
├── __init__.py
├── losses.py
├── metrics.py
├── blocks.py
├── features.py
├── backbones.py
├── adapters.py
├── necks.py
└── heads.py
```

## 4. import 재작성 규칙

**intra(같은 파일로 병합됨) → 삭제**
- `from src.losses.base_loss import BaseLoss`
- `from src.metrics.base_metric import BaseMetric`
- `from src.models.backbones.base_backbone import BaseBackbone`
- `from src.models.adapters.base_adapter import BaseBackboneAdapter`
- `from src.models.blocks.conv_block import ConvBlock` (단, `blocks.py` 내부에서만)

**cross(다른 component로 유지) → 새 경로**
- `from src.models.blocks.conv_block import ConvBlock` → `from src.components.blocks import ConvBlock`
  (사용처: `backbones.py`, `necks.py`, `heads.py`)
- `from src.models.blocks.deconv_block import DeconvBlock` → `from src.components.blocks import DeconvBlock`
  (사용처: `necks.py`)
- `from src.models.features import FeatureBundle` → `from src.components.features import FeatureBundle`
  (사용처: `adapters.py`)

**evaluator.py 갱신** — [../../src/core/evaluator.py](../../src/core/evaluator.py) 8–10행:
```python
from src.components.metrics import MeanCornerDistance, MaxCornerDistance, PCK
from src.components.metrics import PolygonIoU
from src.components.metrics import SuccessRate
```

## 5. 이름 충돌 확인

대상 8개 모듈에 병합된 최상위 정의는 도메인 내 중복이 없다.
- losses: `BaseLoss`, `BCELoss`, `DiceLoss`, `FocalLoss`, `HeatmapMSELoss`, `SmoothL1Loss`, `WingLoss`
- metrics: `BaseMetric`, `CornerDistanceMetric`, `MeanCornerDistance`, `MaxCornerDistance`, `PCK`, `PolygonIoU`, `SuccessRate`
- blocks: `ConvBlock`, `DeconvBlock` / features: `FeatureBundle`, `FeatureSpec`, `FeatureExtractor`
- backbones: `BaseBackbone`, `CustomBackbone`, `TimmBackbone`, `TorchBackbone`(+상수)
- adapters: `BaseBackboneAdapter`, `CNNBackboneAdapter`, `TransformerBackboneAdapter`
- necks: `MultiScaleNeck` / heads: `CoordGapHead`, `CoordSpatialHead`, `DetectionHead`, `HeatmapHead`, `MaskHead`(+`NUM_CORNER_CLASSES`)

## 6. 미해결 의존 (후속 plan 대상)

- `src/models/reg|seg|heatmap|det`의 model/wrapper는 아직 ver3에 없다. 이 파일들이 이관될 때
  `from src.models.blocks...` / `from src.models.features...` / `from src.losses...` /
  `from src.metrics...` import를 모두 `from src.components.*`로 재작성한다(§4의 규칙 재사용).
- `src/core/factory.py`의 지연 import(`src.models.*`)는 모델 이관 plan에서 해소한다(0003 §4와 동일).

## 7. 완료 기준

- `src/components/`에 `__init__.py`와 8개 flatten 모듈이 존재하고, 각 모듈이 대응 ver2 원본의
  모든 최상위 정의/상수를 담을 것 — 충족
- 병합 파일 내 intra import가 남아있지 않고, cross import는 전부 `src.components.*` 경로일 것 — 충족
- `src/core/evaluator.py`가 `src.components.metrics`를 참조할 것 — 충족
- ver3에 `src/losses/`, `src/metrics/`, `src/models/`(building-block 부분)는 새로 만들지 않을 것 — 충족
- ver2 파일은 하나도 수정되지 않을 것 — 충족

## 8. 검증

실행 환경은 conda `pytorch_env`(ver2 CLAUDE.md §5)를 사용한다.

- **컴파일**: `python -m py_compile src/components/*.py src/core/evaluator.py` → 전체 통과.
- **import 스모크**:
  ```bash
  conda activate pytorch_env
  cd <ver3>
  python -c "from src.components import losses, metrics, blocks, features"   # OK
  python -c "from src.components import backbones, adapters, necks, heads"    # OK (timm/torchvision 로드)
  python -c "import src.core.evaluator"                                       # OK (metrics 경로 갱신 확인)
  ```
  세 명령 모두 통과 확인.
- **잔여 참조 없음**:
  ```bash
  grep -rn --include=*.py -E "src\.(losses|metrics)\b|src\.models\.(blocks|features|backbones|adapters|necks|heads)" src/
  ```
  결과 없음 확인(components 내부·evaluator 모두 새 경로 사용).
