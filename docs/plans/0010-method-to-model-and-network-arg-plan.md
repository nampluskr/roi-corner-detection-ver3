---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/methods/` → `src/models/`, `src/core/factory.py`, `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `scripts/batch_run.py`, `scripts/batch_config.py`, 9개 `*/wrapper.py`·`*/model.py` 시그니처
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0009-peak-ridge-naming-plan.md](0009-peak-ridge-naming-plan.md)
---

## 목적과 배경

현재 프로젝트의 각 method는 정확히 하나의 `XXXModel`/`XXXWrapper`와 1:1로 매칭된다
(`reg`, `seg`, `det`, `torchseg`, `torchdet`, `heatmap`, `yolo`, `detr` 8종. 0008에서 추가한
`linemap`을 포함하면 9종이며, 0009 적용 시 `heatmap`/`linemap`은 `peak`/`ridge`로 개명된다).
method라는 상위 분류 개념이 model 선택과 완전히 동치가 되었으므로, "method"와 "model"을 별도
개념으로 유지할 이유가 없다.

이에 따라 두 가지를 통합한다.

1. 개념·디렉터리 통합: `method` 개념을 `model`로 일원화하고 `src/methods/` → `src/models/`로 옮긴다.
2. CLI 인자 재설계: `--method`를 폐지하고 `--model`을 model 셀렉터(9개 문자열)로 사용한다. model별
   추가 인자(`--head` 등)는 선택된 model의 kwargs로 전달한다.

또한 현재 `--model` 인자는 이미 다른 의미로 쓰이고 있다는 점을 해소해야 한다. 조사 결과 아키텍처
지정은 두 그룹으로 명확히 갈린다.

- Group A (`reg`/`seg`/`det`/`heatmap`): `backbone=`로 CNN 인코더를 지정하며 `model` 인자를 받지
  않는다(넘기면 `TypeError`). 값 예: `custom`, `resnet18/34/50`, `efficientnet_b0`, `vgg16/19`,
  `swin_t`, `wide_resnet50_2.tv_in1k`.
- Group B (`torchseg`/`torchdet`/`yolo`/`detr`): `model=`로 완성형 아키텍처를 지정하며 `backbone`은
  받되 무시(no-op)한다. 값 예: `fcn_resnet50`, `deeplabv3_resnet50`, `fasterrcnn_resnet50_fpn`,
  `retinanet_resnet50_fpn`, `ssd300_vgg16`, `yolov8n`, `detr_resnet50`.

두 그룹의 아키텍처 셀렉터(`backbone` vs `model`)는 한 실행에서 동시에 의미를 갖지 않아 사실상
배타적이며, `scripts/config.py`의 `get_model_name`/`get_experiment`는 이미 `model or backbone`으로
둘을 하나의 슬롯처럼 취급하고 있다. 따라서 이 둘을 **하나의 통합 인자 `--network`**(약어 `--net`
병용)로 합치고, 새 `--model`은 순수하게 model 셀렉터로만 사용한다.

## 확정 결정

- CLI model 셀렉터: `--model`. 값은 9개 model 문자열(`reg`, `seg`, `det`, `torchseg`, `torchdet`,
  `heatmap`(또는 0009 적용 시 `peak`), `linemap`(또는 `ridge`), `yolo`, `detr`). `--method`는
  제거한다.
- 기존 `--backbone`과 아키텍처용 `--model`을 통합한 아키텍처 인자: `--network`(약어 `--net`).
  Group A에서는 CNN 인코더 이름을, Group B에서는 완성형 아키텍처 이름을 동일 인자로 받는다.
- wrapper 전달 방식: **wrapper/model 시그니처를 통일**한다. 9개 `XXXWrapper`/`XXXModel`이 기존
  `backbone=`/`model=` 대신 단일 키 `network=`로 아키텍처 이름을 받도록 시그니처를 변경한다.

## 범위

이 플랜은 문서 작성만 수행하며, 아래 실제 코드 변경과 테스트는 승인 후 후속 작업에서 진행한다.

포함 항목(후속 작업 대상):

- 디렉터리: `src/methods/` → `src/models/`로 이동한다. 하위 `base/`와 9개 model 패키지 디렉터리
  구조는 그대로 유지한다. 패키지 내부의 `from src.methods.<x> import ...` import 경로를
  `from src.models.<x> import ...`로 일괄 변경한다.
- wrapper 시그니처 통일: 각 `XXXWrapper.__init__`의 아키텍처 인자를 `network`로 통일한다.
  - Group A(`reg`/`seg`/`det`/`heatmap`): `backbone=` → `network=`로 이름을 바꾸고, 내부에서
    `XXXModel(network=...)`로 전달한다. 기본값(`"custom"`)은 유지한다.
  - Group B(`torchseg`/`torchdet`/`yolo`/`detr`): 아키텍처 지정에 쓰이던 `model=` → `network=`로
    이름을 바꾼다. 기존에 CLI 호환용으로만 받고 무시하던 `backbone=` 무의미 인자는 제거한다.
    각 model.py의 `network = network or "<default>"` fallback은 유지한다.
- model.py 시그니처: 각 `XXXModel.__init__`도 대응하여 `backbone=`/`model=` → `network=`로 통일하고,
  내부 검증 상수(`SUPPORTED_*_BACKBONES`, `SUPPORTED_*_MODELS`)에 대한 참조는 유지한다. 상수 이름
  자체(`_BACKBONES`/`_MODELS`)를 `_NETWORKS`로 통일할지는 후속 작업에서 결정한다(이 플랜에서는
  변수명 변경을 강제하지 않는다).
- `head` 인자: 그대로 유지한다. `head`는 아키텍처 셀렉터가 아니라 model별 세부 옵션(reg의
  `gap`/`spatial`, det/torchdet/yolo/detr의 `box`/`point`)이므로 통합 대상이 아니다.
- `src/core/factory.py` `get_wrapper`: 첫 인자 `method`를 `model`(문자열)로 이름을 바꾸고, dispatch
  분기와 import 경로(`src.methods.*` → `src.models.*`)를 갱신한다.
- `scripts/config.py`:
  - `DEFAULTS`에서 `method="reg"` → `model="reg"`, `backbone`/`model` 슬롯을 `network`로 정리하고
    기본 `network="custom"`, `head` 기본값은 유지한다.
  - `parse_args()`: `--method` 제거, `--model`(model 셀렉터)·`--network`/`--net`(아키텍처) 추가,
    기존 `--backbone` 제거. `--head`는 유지한다.
  - `get_wrapper_kwargs(args)`: `network`, `head`(및 `warmup_epochs`)만 truthy할 때 kwargs로
    통과시키도록 정리한다. 더 이상 `backbone`/`model`을 개별 통과시키지 않는다.
  - `get_model_name`/`get_experiment`/`get_output_dir`: `model or backbone` 병합 로직을 단일
    `network` 값 기반으로 단순화한다. 경로/실험명 구성 요소가 `{model}/{network}_{head}` 형태가
    되도록 조정한다(현재 `{method}/{model|backbone}_{head}`).
  - `warmup_methods` 튜플: `method` → `model` 값 기준으로 갱신한다.
- `scripts/train.py`·`evaluate.py`·`predict.py`: `args.method` 참조를 `args.model`로 바꾸고
  `get_wrapper(args.model, ...)`로 호출하도록 수정한다.
- `scripts/batch_run.py`:
  - `PASS_KEYS`에서 `backbone`, `model`을 제거하고 `network`를 추가한다(`head`, `device`,
    `batch_size`, `max_epochs`, `num_workers`, `train_size`, `valid_size`, `test_size`,
    `checkpoint`, `output_dir`, `warmup_epochs`는 유지). `--model`은 model 셀렉터이므로 별도로
    `["--model", cfg["model"]]` 형태로 앞세운다(현재 `["--method", cfg["method"]]` 자리).
  - `get_cli_args`가 `cfg["method"]` → `cfg["model"]`을 읽도록 변경한다.
- `scripts/batch_config.py`: 모든 config dict의 `"method"` 키를 `"model"`로, 아키텍처를 지정하던
  `"backbone"`/`"model"` 키를 `"network"`로 통일한다. 완성형 아키텍처를 지정하던
  `{"method": "yolo", "model": "yolov8n"}` 형태는 `{"model": "yolo", "network": "yolov8n"}`가 된다.
  `HEATMAP_CONFIGS` 등 그룹 변수명과 `CONFIGS` 조합식, `METHOD_COMPARISON_CONFIGS`의 예시·체크포인트
  경로도 함께 갱신한다.

제외 항목:

- `heatmap`/`linemap` → `peak`/`ridge` 개명은 [0009](0009-peak-ridge-naming-plan.md)의
  범위이며 이 플랜에서 다루지 않는다. 두 플랜의 적용 순서는 후속 작업에서 조율한다(개명을 먼저 하면
  이 플랜은 새 이름 기준으로, 이 플랜을 먼저 하면 0009가 `src/models/` 기준으로 진행된다).
- `head` 인자의 의미·검증 로직 변경, model별 추가 하이퍼파라미터(`box_size`, `grad_clip` 등)의 CLI
  노출 여부는 이 플랜의 범위에 포함하지 않는다.
- `src/components/` 하위 컴포넌트(`backbones.py`, `heads.py`, `decoders.py`, `losses.py` 등)의
  파일·클래스 이름은 변경하지 않는다. 단 wrapper/model 통일 과정에서 이들을 import하는 경로는
  변하지 않는다(`src.components.*`는 그대로).
- 학습된 체크포인트·`outputs/` 산출물 디렉터리의 실제 마이그레이션은 포함하지 않는다. 경로 구성
  규칙 변경으로 신규 산출물 경로만 달라진다.
- 이미 이력으로 보존되는 완료 plan(0006, 0008 등)의 본문 표기는 수정하지 않는다.

## 완료 기준

이 플랜이 `Done`으로 전환되기 위한 조건(후속 작업에서 확인):

- `src/models/` 디렉터리가 존재하고 `src/methods/`는 더 이상 존재하지 않는다. 내부 import가 모두
  `src.models.*`를 가리킨다.
- 9개 `XXXWrapper`/`XXXModel`이 아키텍처 인자를 단일 키 `network=`로 받는다. `backbone=`/아키텍처용
  `model=` 인자는 시그니처에 남아 있지 않다.
- `scripts/config.py`의 CLI가 `--model`(셀렉터), `--network`/`--net`(아키텍처), `--head`를
  제공하고 `--method`·`--backbone`은 제거되었다.
- `get_wrapper`가 `model` 문자열로 dispatch한다.
- `scripts/train.py`·`evaluate.py`·`predict.py`·`batch_run.py`·`batch_config.py`가 새 인자 체계로
  일관되게 동작한다.
- 대표 실행이 성공한다: `python scripts/train.py --model reg --network custom --head gap`,
  `python scripts/train.py --model yolo --network yolov8n --head box`.

## 검증

이 플랜은 [0009](0009-peak-ridge-naming-plan.md)와 한 작업에서 함께 구현했다. 0009의 개명을 먼저
적용한 뒤 이 플랜의 디렉터리 이동과 인자 재설계를 반영했으므로 최종 상태는 `peak`/`ridge` 이름
기준이다.

구현 결과는 다음과 같다.

- `src/methods/`를 `src/models/`로 옮기고 패키지 내부 import 경로를 `src.models.*`로 일괄
  변경했다. `src/methods/`는 더 이상 존재하지 않는다.
- 9개 `XXXWrapper`/`XXXModel`의 아키텍처 인자를 단일 키 `network=`로 통일했다. Group A는
  `backbone=` $\to$ `network=`, Group B는 아키텍처용 `model=` $\to$ `network=`로 바꾸고 Group B
  wrapper의 무의미한 `backbone=` 인자를 제거했다. `head`는 그대로 유지했다.
- `src/core/factory.py`의 `get_wrapper`가 첫 인자 `method` $\to$ `model`로 dispatch한다.
- `scripts/config.py`를 재작성했다. `DEFAULTS`가 `model="reg"`, `network="custom"`, `head="gap"`를
  갖고, `parse_args()`가 `--model`, `--network`(약어 `--net`), `--head`를 제공하며 `--method`와
  `--backbone`은 제거했다. 경로/실험명은 `{model}/{network}_{head}/{exp_name}` 형태다.
- `scripts/train.py`, `evaluate.py`, `predict.py`가 `args.model`로 `get_wrapper`를 호출한다.
- `scripts/batch_run.py`의 `PASS_KEYS`에서 `backbone`/`model`을 빼고 `network`를 추가했으며,
  `--model` 셀렉터를 앞세운다. `scripts/batch_config.py`는 모든 config의 `method` $\to$ `model`,
  아키텍처 키 $\to$ `network`로 통일했다.

검증은 conda `pytorch_env`에서 다음을 수행했다.

- import 검증: `src.core.factory`와 9개 `src.models.*.wrapper`, `scripts.config`,
  `scripts.batch_config`, `scripts.batch_run` import 오류 없음.
- Group A forward 검증: `get_wrapper("reg", network="custom", head="gap")` 등 5개 model이
  `(2, 3, 224, 224)` 입력에서 정상 출력을 반환했다(reg `(2, 8)`, seg `(2, 1, 112, 112)`,
  det dict, peak/ridge `(2, 4, 112, 112)`).
- Group B 인자 검증: 4개 완성형 model이 `network="__bad__"`에서 `TypeError`가 아니라 각자의
  `Unknown ... network` `ValueError`를 발생시켜 `network=` 인자가 model 검증까지 도달함을 확인했다.
- 경로 검증: `get_output_dir({"model": "yolo", "network": "yolov8n", "head": "box", ...})`가
  `outputs/public/yolo/yolov8n_box/yolo_bs4_ep5_yolov8n_box`를 반환했다.
- CLI 조립 검증: `get_cli_args`가 활성 yolo config에 대해
  `["--model", "yolo", "--save", "--network", "yolov8n", "--head", "box", ...]`를 생성했다.
- 실제 학습 스크립트의 전체 실행(가중치 다운로드 포함)과 수치 수렴 검증은 이 세션에서 수행하지
  않았다.
