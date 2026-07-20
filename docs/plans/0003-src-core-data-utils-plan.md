# src/core, src/data, src/utils 원본 이관

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-20 |
| 적용 범위 | ver3 `src/__init__.py`, `src/core/`, `src/data/`, `src/utils/` |
| 관련 문서 | [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0002-experiments-into-scripts-plan.md](0002-experiments-into-scripts-plan.md) |

## 1. 목적과 배경

이 프로젝트의 주된 재구성 대상은 ver2의 `XXXModel`/`XXXWrapper`와 그 관련 구성요소
(`src/models/`, 및 그 의존인 `src/losses/`, `src/metrics/`)다. 그 외 코드는 재구성하지 않고
ver2 원본을 그대로 이관한다.

`src/core/`(factory/trainer/evaluator/predictor), `src/data/`(dataloader/dataset/transforms),
`src/utils/`(geometry/io)는 재구성 대상이 아니므로 내용 변경 없이 ver3로 복사해 실행 기반을
갖춘다. [0002](0002-experiments-into-scripts-plan.md)에서 이관한 `scripts/*.py`가 이들을
import 하므로, 이 이관으로 `scripts/` 실행 그래프의 비-재구성 의존부가 채워진다.

ver2 내부 파일은 참고·읽기 전용이므로([../CLAUDE.md](../CLAUDE.md)), 결과는 ver3 내부에 생성한다.
ver2에는 어떤 변경도 가하지 않는다.

## 2. 범위

포함:
- `src/__init__.py` (패키지 루트) 그대로 이관
- `src/core/` 그대로 이관 — `__init__.py`, `evaluator.py`, `factory.py`, `predictor.py`,
  `trainer.py`
- `src/data/` 그대로 이관 — `__init__.py`, `dataloader.py`, `dataset.py`, `transforms.py`
- `src/utils/` 그대로 이관 — `__init__.py`, `geometry.py`, `io.py`
- `__pycache__` 등 컴파일 산출물은 이관하지 않는다

제외 (후속 plan에서 수행 — 재구성 대상):
- `src/models/` (`XXXModel`/`XXXWrapper`) — 이 프로젝트의 주 재구성 대상
- `src/losses/`, `src/metrics/` — 모델 재구성과 함께 정리

## 3. 재구성 매핑

| ver2 원본 | ver3 신규 | 변경 내용 |
| --- | --- | --- |
| `src/__init__.py` | `src/__init__.py` | 그대로 |
| `src/core/*.py` | `src/core/*.py` | 그대로 (바이트 동일) |
| `src/data/*.py` | `src/data/*.py` | 그대로 (바이트 동일) |
| `src/utils/*.py` | `src/utils/*.py` | 그대로 (바이트 동일) |

이관 후 ver3 `src/` 구조는 다음과 같다.

```text
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── factory.py
│   ├── predictor.py
│   └── trainer.py
├── data/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset.py
│   └── transforms.py
└── utils/
    ├── __init__.py
    ├── geometry.py
    └── io.py
```

## 4. 미해결 의존 (후속 plan 대상)

이관한 `core/`는 재구성 대상 모듈을 아직 없는 상태로 참조한다. 이는 의도된 것이며, 재구성 대상을
이관·재설계하는 후속 plan에서 해소한다.

- `core/factory.py` — 함수 내부 지연 import로 `src.models.reg.wrapper.RegWrapper`,
  `src.models.seg.wrapper.SegWrapper`, `src.models.heatmap.wrapper.HeatmapWrapper`,
  `src.models.det.wrapper.*`, `src.models.det.model.*`를 참조한다. 지연 import이므로 모듈 로드
  시점에는 실패하지 않고, `get_wrapper()` 호출 시에만 `src/models/`가 필요하다.
- `core/evaluator.py` — 파일 상단에서 `src.metrics.corner_distance`, `src.metrics.polygon_iou`,
  `src.metrics.success_rate`를 import 한다. 따라서 `evaluator.py`를 import 하려면 `src/metrics/`
  이관이 선행되어야 한다.

## 5. 완료 기준

- ver3 `src/`에 `__init__.py`와 `core/`, `data/`, `utils/` 하위 `.py` 파일이 ver2와 바이트
  동일하게 존재할 것
- `__pycache__`가 이관되지 않을 것
- ver2 파일은 하나도 수정되지 않을 것

## 6. 검증

`src/models/`, `src/metrics/`가 아직 없으므로 `core/evaluator.py`를 포함한 전체 import 실행
검증은 이 단계에서 불가능하다. 이관은 텍스트 동등성으로 검증한다.

- `src/__init__.py`, `src/core/*.py`, `src/data/*.py`, `src/utils/*.py`가 각각 ver2 원본과
  `diff` 결과 무차이임을 확인
- ver3 `src/`에 `__pycache__`가 없음을 확인
