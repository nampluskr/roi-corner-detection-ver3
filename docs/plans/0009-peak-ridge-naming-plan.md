---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/methods/heatmap/` → `src/methods/peak/`, `src/methods/linemap/` → `src/methods/ridge/`, `src/components/heads.py`, `src/core/factory.py`, `scripts/config.py`, `scripts/batch_config.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md), [0006-heatmap-postprocessor-argmax-plan.md](0006-heatmap-postprocessor-argmax-plan.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0010-method-to-model-and-network-arg-plan.md](0010-method-to-model-and-network-arg-plan.md)
---

## 목적과 배경

`heatmap` method는 코너를 점(point) 중심의 가우시안 피크(Gaussian peak)로 표현하고, `linemap`
method는 인접한 두 코너를 지나는 무한 직선을 가우시안 능선(Gaussian ridge)으로 표현한다. 두 방식이
공존하게 되면서 각 표현 형태를 더 정확히 반영하는 이름이 필요하다는 문제 제기가 있었다.

명명 검토는 두 단계로 진행되었다.

- 1차 검토에서는 "산" 비유(peak/ridge)를 살린 `peakmap`/`ridgemap`을 후보로 삼았다.
- 이후 기존 model 이름들(`reg`, `seg`, `det`)이 `-map` 같은 접미어 없이 개념을 간결하게 지칭하는
  패턴임을 반영해, 접미어 `-map`을 떼고 `peak`/`ridge`로 축약하는 것으로 확정한다.

`heatmap` → `peak`, `linemap` → `ridge`로 개명한다. 근거는 다음과 같다.

- 기존 짧은 model 이름(`reg`/`seg`/`det`)과 길이·톤이 일치하며, 접미어 없는 개념어 계열로 전체
  이름 체계가 통일된다. `-map`은 dense map 출력이라는 공통 속성이므로 이름에 포함하지 않아도 의미가
  충분하다.
- `peak`(가우시안 피크 = 점 표현), `ridge`(가우시안 능선 = 선 표현)만으로 표현 방식을 명확히
  지칭할 수 있다.
- 중간 이름 `peakmap`/`ridgemap`을 거치지 않고 곧바로 `peak`/`ridge`로 확정하여, 개명을 한 번에
  끝낸다. 이 플랜은 앞서 검토하던 `peakmap`/`ridgemap` 안을 대체한다.

## 범위

이 플랜은 문서 작성만 수행하며, 아래 항목의 실제 코드 변경과 테스트는 후속 작업에서 진행한다.

포함 항목(후속 작업에서 변경할 대상):

- `src/methods/heatmap/` 디렉터리를 `src/methods/peak/`로 이름을 바꾸고, 내부 클래스명을 변경한다:
  `HeatmapModel` → `PeakModel`, `HeatmapPreprocessor` → `PeakPreprocessor`, `HeatmapPostprocessor`
  → `PeakPostprocessor`, `HeatmapWrapper` → `PeakWrapper`. 모델 속성 `heatmap_stride` →
  `peak_stride`도 함께 변경한다.
- `src/methods/linemap/` → `src/methods/ridge/` 개명은 **이미 완료되었다**. 디렉터리 이동과 함께
  클래스명(`LinemapModel` → `RidgeModel`, `LinemapPreprocessor` → `RidgePreprocessor`,
  `LinemapPostprocessor` → `RidgePostprocessor`, `LinemapWrapper` → `RidgeWrapper`), 모델 속성
  (`linemap_stride` → `ridge_stride`), preprocessor 인자(`linemap_size` → `ridge_size`), head 문자열
  (`"linemap"` → `"ridge"`)까지 반영했다. 남은 `peak` 쪽(아래 항목)과 factory/스크립트 연결만
  후속 작업에서 처리하면 된다.
- 이름 축약으로 인한 식별자 충돌 점검: `ridge` postprocessor는 능선의 정점(peak)과 방향을 다루므로
  지역 변수·주석에서 `peak`/`ridge`가 클래스명·문자열과 혼용될 수 있다. 개명 시 클래스/디렉터리
  이름과 알고리즘 설명용 일반 명사가 코드 상에서 명확히 구분되는지 확인한다.
- `src/components/heads.py`의 `HeatmapHead`: 현재 `heatmap`(→`peak`)과 `linemap`(→`ridge`) 두
  모델이 동일한 4채널 1x1 conv 헤드를 공유한다. 이름을 두 model 어느 쪽에도 편중되지 않는
  형태(예: `FourChannelDenseHead`)로 정리할지, 아니면 유지 후 docstring만 갱신할지, 또는
  `peak`/`ridge` 전용으로 분리할지를 후속 작업에서 결정한다. 이 플랜에서는 결정을 확정하지 않는다.
- head 이름 문자열: `PeakWrapper`/`PeakModel`에서 사용하는 head 문자열 `"heatmap"` → `"peak"`,
  `RidgeWrapper`/`RidgeModel`에서 사용하는 head 문자열 `"linemap"` → `"ridge"`로 변경한다
  (`wrapper.py`의 `head not in (None, "...")` 검증 및 에러 메시지 포함).
- `src/core/factory.py`의 `get_wrapper`: `method == "heatmap"` 분기를 `method == "peak"`으로
  변경하고 `PeakWrapper`를 import하도록 수정한다. 아직 연결되지 않은 `linemap`/`ridge` 분기를
  `method == "ridge"`로 신규 추가하고 `RidgeWrapper`를 import하도록 한다.
- `scripts/config.py`의 `warmup_methods` 튜플: `"heatmap"` → `"peak"`으로 변경하고 `"ridge"`를
  추가한다.
- `scripts/batch_config.py`: `HEATMAP_CONFIGS` 변수명과 내부 `"method": "heatmap"`,
  `"head": "heatmap"` 항목을 `PEAK_CONFIGS`/`"method": "peak"`/`"head": "peak"`으로 변경한다.
  `CONFIGS` 조합식과 `METHOD_COMPARISON_CONFIGS`의 `"method": "heatmap"` 예시 항목, 체크포인트
  경로 예시(`outputs/public/heatmap/custom_heatmap/...`)도 함께 갱신한다. `ridge` 실험
  조합(`RIDGE_CONFIGS`) 추가 여부는 후속 작업에서 결정한다.

제외 항목:

- `src/components/decoders.py`(`UNetDecoder`), `src/components/losses.py`(`HeatmapFocalLoss`,
  `HeatmapMSELoss`)의 이름은 이 플랜에서 변경하지 않는다. `Heatmap` 접두어가 붙어 있지만 다른
  method(`seg` 등)에서도 재사용 가능한 일반 컴포넌트이므로, 이름 변경 여부는 별도 검토가 필요하다.
- 이미 완료되어 이력으로 보존되는 [0006](0006-heatmap-postprocessor-argmax-plan.md) 문서 본문의
  `heatmap` 표기는 과거 기록이므로 수정하지 않는다. [0008](0008-ridge-method-plan.md)은 `linemap`
  도입 기록이지만 `ridge` 개명이 완료되어 해당 문서 본문은 `ridge` 표기로 이미 갱신되었다.
- 학습된 체크포인트 파일, `outputs/` 하위 산출물 디렉터리의 실제 이름 변경(마이그레이션)은 이
  플랜의 범위에 포함하지 않는다.
- 다른 method(`reg`, `seg`, `det`, `torchseg`, `torchdet`, `yolo`, `detr`)는 변경하지 않는다.
- [0010](0010-method-to-model-and-network-arg-plan.md)의 `src/methods/` → `src/models/` 이동 및
  CLI 인자(`--model`/`--network`) 통합은 별도 플랜이다. 두 플랜의 적용 순서는 후속 작업에서
  조율한다. 개명(0009)을 먼저 하면 0010은 `peak`/`ridge` 기준으로, 0010을 먼저 하면 0009는
  `src/models/` 기준으로 진행된다.

## 완료 기준

이 플랜이 `Done`으로 전환되기 위한 조건(후속 작업에서 확인):

- `src/methods/peak/`, `src/methods/ridge/` 디렉터리가 각각 존재하고, 클래스명·속성명이 위 범위대로
  변경되어 있다.
- `src/core/factory.py`의 `get_wrapper`가 `"peak"`, `"ridge"` 두 method 문자열을 모두 처리한다.
- `scripts/config.py`, `scripts/batch_config.py`가 `"heatmap"`/`"linemap"` 문자열을 더 이상
  참조하지 않는다(이력 문서 제외).
- `src/components/heads.py`의 공유 헤드 이름에 대한 결정이 확정되고 반영되어 있다.
- `PYTHONPATH=<project-root> python -c "import src.core.factory; import src.methods.peak.wrapper;
  import src.methods.ridge.wrapper"` 등으로 import 오류가 없음을 확인한다.

## 검증

이 플랜은 [0010](0010-method-to-model-and-network-arg-plan.md)과 한 작업에서 함께 구현했다. 개명을
먼저 적용하고 이어서 0010의 `src/methods/` $\to$ `src/models/` 이동과 `--model`/`--network` 인자
체계를 반영했으므로, 최종 상태는 `src/models/peak/`, `src/models/ridge/` 기준이다.

구현 결과는 다음과 같다.

- `src/methods/heatmap/`를 `src/models/peak/`로 옮기고 클래스명을 `PeakModel`,
  `PeakPreprocessor`, `PeakPostprocessor`, `PeakWrapper`로, 모델 속성을 `peak_stride`로,
  preprocessor 인자를 `peak_size`로, head 문자열을 `"peak"`로 변경했다.
- `src/components/heads.py`의 공유 헤드는 `HeatmapHead` $\to$ `FourChannelDenseHead`로 개명하여
  `peak`/`ridge` 어느 쪽에도 편중되지 않도록 정리했다. `peak`과 `ridge`의 model.py가 이 이름을
  import한다.
- `src/components/losses.py`의 `HeatmapFocalLoss`/`HeatmapMSELoss`는 제외 항목대로 유지했다.
- `src/core/factory.py`의 `get_wrapper`가 `"peak"`, `"ridge"` 분기를 처리한다(0010에서 첫 인자가
  `method` $\to$ `model`로 함께 바뀌었다).
- `scripts/config.py`의 warmup 목록과 `scripts/batch_config.py`의 `PEAK_CONFIGS`(및 신규
  `RIDGE_CONFIGS`), `METHOD_COMPARISON_CONFIGS` 예시와 체크포인트 경로를 갱신했다.

검증은 conda `pytorch_env`에서 다음을 수행했다.

- import 검증: `PYTHONPATH=<project-root> python -c "import src.core.factory; import
  src.models.peak.wrapper; import src.models.ridge.wrapper"` 오류 없음.
- forward 검증: `get_wrapper("peak", network="custom")`, `get_wrapper("ridge", network="custom")`가
  `(2, 3, 224, 224)` 입력에 대해 `(2, 4, 112, 112)` 출력을 반환함을 확인했다.
- 실제 학습 스크립트의 전체 실행과 수치 수렴 검증은 이 세션에서 수행하지 않았다.
