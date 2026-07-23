# 0022 offset model plan

이 문서는 ver1의 homography regression model을 ver3의 통일된 model pattern으로 이식하여 `offset`
이라는 새 model로 추가하는 작업을 기록한다.

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `src/models/offset/`, `src/core/factory.py`, `scripts/config.py`, `README.md`, `docs/models/`, `docs/README.md`, `slides/` |
| 관련 문서 | [0010-method-to-model-and-network-arg-plan.md](0010-method-to-model-and-network-arg-plan.md) |

## 1. 목적과 배경

ver1(`260701_roi-corner-detection-ver1`)에는 homography regression model(`src/models/homography/`)이
있으며 이것이 기존 방법론이다. 이 model은 backbone과 spatial 정보를 보존하는 strided-conv head로
구성되어 고정된 canonical square의 네 vertex에 대한 offset 8개를 회귀한다. 이는 DeTone 계열의
4-point parametrization으로 homography와 정보량이 동등하며, 새로운 localization 방법을 비교하는
저비용 regression baseline 역할을 한다.

ver3에는 아직 이 model이 이식되지 않았다. 이 작업의 목표는 해당 model을 `offset`이라는 이름의 새
model로 가져오되 ver1 code를 그대로 복제하지 않고 ver3의 통일된 model pattern을 따르는 것이다.
ver3의 `reg` model은 이미 `CoordSpatialHead`(ver1 homography head와 동일)와 통일된 backbone stack을
재사용하므로 `offset`은 사실상 canonical-offset parametrization만 다른 `reg`다. 실제로 새로 작성하는
logic은 preprocessor에서 canonical square를 빼는 것, postprocessor에서 `alpha*tanh`와 canonical
square를 더하는 것, loss 경로를 postprocessor와 일치시키는 `compute_losses` override뿐이다.

## 2. 범위

포함 항목은 다음과 같다.

- 새 model package `src/models/offset/`: `__init__.py`(비어 있음), `model.py`, `preprocessor.py`,
  `postprocessor.py`, `wrapper.py`.
- `offset`을 factory(`src/core/factory.py`)와 CLI config(`scripts/config.py`)에 등록.
- canonical 문서 갱신: `README.md` model table과 model 개수, `docs/models/08-offset.md` 신설,
  `docs/models/README.md`와 `docs/README.md`의 색인 반영.
- slides 반영: `slides/outline.md`의 관련 page 갱신과 offset 전용 page 신설, `slides/README.md`의
  asset 목록 추가.

제외 항목은 다음과 같다.

- ver1과 ver2는 수정하지 않는다. 읽기 전용 참조만 사용한다.
- reprojection auxiliary loss와 `estimate_homography`는 이식하지 않는다. ver3에는
  reprojection-error metric이 없고 ver1 homography wrapper도 이를 사용하지 않았으므로 offset에 대한
  단순 smooth L1로 유지한다.
- 아래 smoke 검증 이외의 training run은 수행하지 않는다.
- slides asset 이미지(`assets/postprocess_offset.png`) 자동 생성은 하지 않는다. page는 참조만
  추가한다.

## 3. 설계

ver1 구성 요소를 ver3 구성 요소로 다음과 같이 대응한다.

| ver1 homography | ver3 offset |
| --- | --- |
| `HomographyModel`: ResNet hardcoding과 inline strided-conv head | `OffsetModel(BaseModel)`: `CustomBackbone`/`TorchBackbone`/`TimmBackbone` + `CoordSpatialHead` 재사용, `reg`처럼 `--network`/`--head`로 선택 |
| `--backbone` selector | `--network`/`--net` selector와 `--head spatial` |
| 수동 discriminative-LR AdamW | ver3 two-phase warmup `build_optimizer`/`build_scheduler` |
| `CANONICAL_CORNERS`, `MARGIN`, `ALPHA` module 상수 | `offset/model.py`에 동일 상수 정의 |
| `compute_losses`에서 `ALPHA*tanh` 적용 | 동일 override |

`src/models/offset/model.py`는 `src/models/reg/model.py`의 `_build_extractor_and_head` 구조를
재사용한다. 기본값은 `head="spatial"`이며 `reg` parity를 위해 `gap`도 지원한다. module 수준에
`MARGIN = 0.25`, `ALPHA = 0.25`, `CANONICAL_CORNERS`(`[[m,m],[1-m,m],[1-m,1-m],[m,1-m]]`,
TL/TR/BR/BL)를 정의하여 preprocessor, postprocessor, wrapper가 한 곳에서 import한다. extractor
submodule은 `self.extractor`로 명명하여 `BaseWrapper.get_backbone_module`과 warmup freeze가
동작하도록 한다. `forward(images)`는 raw `(N, 8)`을 반환한다.

`src/models/offset/preprocessor.py`의 `OffsetPreprocessor(BasePreprocessor)`는 `(N, 4, 2)` corners에서
`CANONICAL_CORNERS`를 빼고 `(N, 8)` offset target으로 reshape한다.

`src/models/offset/postprocessor.py`의 `OffsetPostprocessor(BasePostprocessor)`는
`offsets = ALPHA * tanh(raw)`를 적용하고 `(N, 4, 2)`로 reshape한 뒤 canonical vertex를 더하고
`clamp(0, 1)`한다.

`src/models/offset/wrapper.py`의 `OffsetWrapper(BaseWrapper)`는 생성자 signature를 `RegWrapper`와
동일하게 하되 기본값 `head="spatial"`을 사용한다. `build_optimizer`/`build_scheduler`는 `RegWrapper`의
two-phase discriminative-LR warmup을 그대로 사용한다. 기본 loss는 offset 전용 smooth L1이다. ver3의
공용 `SmoothL1Loss`(`src/components/losses.py`)는 dict 입력(`raw_output["box"]`)을 전제로 하는 det
계열 loss이므로 plain `(N, 8)` offset tensor에는 맞지 않는다. 따라서 offset package 안에 plain tensor
smooth L1인 `OffsetSmoothL1Loss(BaseLoss)`를 정의하여 공용 component를 수정하지 않고 ver1의
`SmoothL1Loss()` 의미를 유지한다. metric은 `PolygonIoU()`다. `compute_losses(raw_output, targets)`는
`target = self.preprocessor(targets)`, `offsets = ALPHA * torch.tanh(raw_output)`,
`{name: loss_fn(offsets, target) ...}`을 반환하여 loss 경로를 postprocessor와 일치시킨다.

등록은 다음과 같이 한다. `src/core/factory.py`의 `get_wrapper`에서 마지막 `raise` 앞에 offset 분기를
추가한다. `scripts/config.py`의 `get_wrapper_kwargs`에서 `warmup_models` tuple에 `"offset"`을 추가하여
`--warmup_epochs`가 전달되도록 한다.

## 4. 완료 기준

다음을 모두 충족하면 이 plan을 Done으로 본다.

- `src/models/offset/`의 다섯 파일이 ver3 code 규칙을 따르며 존재한다.
- `--model offset`이 factory에서 `OffsetWrapper`로 dispatch된다.
- canonical 문서와 slides가 offset을 반영하고 model 개수 표기가 12로 갱신된다.
- 아래 검증의 import 확인, factory dispatch 확인, smoke train run이 통과한다.

## 5. 검증

`pytorch_env`에서 ver3 root로 이동해 실행한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

import 확인:

```bash
PYTHONPATH=/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3 \
  python -c "from src.models.offset.wrapper import OffsetWrapper; print('ok')"
```

factory dispatch 확인:

```bash
PYTHONPATH=/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3 \
  python -c "from src.core.factory import get_wrapper; w=get_wrapper('offset', network='custom', head='spatial', warmup_epochs=0); print(type(w).__name__)"
```

smoke train run(작은 subset, 1 epoch):

```bash
python scripts/train.py \
  --dataset public \
  --model offset --network custom --head spatial \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --patience 1 --warmup_epochs 0 \
  --num_workers 0 \
  --output_dir outputs/public/offset/custom_spatial/quickstart \
  --save
```

loss와 IoU가 finite이고 `history.json`, `model.pth`, `run.log`가 생성되면 통과로 본다. 이것은 성능
측정이 아니라 integration smoke test다. 검증 통과 후 이 plan의 상태를 Done으로 갱신한다.
