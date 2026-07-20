# experiments 폴더를 scripts 폴더로 병합

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-20 |
| 적용 범위 | ver3 `scripts/` 폴더 전체 구성 (원본 4파일 이관 + experiments 2파일 리네임 병합) |
| 관련 문서 | [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md) |

## 1. 목적과 배경

ver2는 배치 실험 정의와 실행을 `experiments/` 폴더에 분리해 두고 있다.

- `experiments/configs.py` — 배치 실험 조합(`CONFIGS`)을 정의한다.
- `experiments/run.py` — `subprocess`로 `scripts/<mode>.py`를 반복 실행하는 배치 러너다.

그러나 러너인 `run.py`는 이미 `scripts/config.py`(단일 실험 조회·경로 헬퍼)를 import 하고,
실행 대상 스크립트도 모두 `scripts/`(train/evaluate/predict)에 있다. 즉 배치 실행 관련 코드가
`experiments/`와 `scripts/`로 이원화되어 있고, `experiments/`는 배치 정의·러너 2개 파일만 담고
있다. 이 둘을 `scripts/`로 합쳐 실행 관련 코드를 한 폴더로 모으고 `experiments/` 폴더를 제거한다.

ver2 내부 파일은 참고·읽기 전용이므로([../CLAUDE.md](../CLAUDE.md)), 이 재구성 결과는 ver3
내부에 실제로 생성해 canonical 구조로 확립한다. ver2에는 어떤 변경도 가하지 않는다.

## 2. 범위

포함:
- ver3에 ver2 `scripts/` 원본 4파일을 그대로 이관 — `config.py`, `train.py`, `evaluate.py`,
  `predict.py`. 내용 변경 없이 바이트 동일하게 복사한다.
- ver3에 `scripts/batch_config.py` 생성 — ver2 `experiments/configs.py`의 내용을 그대로 이관.
  파일 헤더 주석 경로만 새 경로로 수정한다.
- ver3에 `scripts/batch_run.py` 생성 — ver2 `experiments/run.py`의 내용을 이관하되 import 한 줄만
  수정한다: `from experiments.configs import CONFIGS` -> `from scripts.batch_config import CONFIGS`.

제외 (후속 plan에서 수행):
- ver2 실제 파일 이동/삭제 (ver2는 read-only)
- `src/` 폴더 이관 — [0003-src-core-data-utils-plan.md](0003-src-core-data-utils-plan.md)에서 수행
- ver2의 완료 plan(0004, 0005, 0007~0011)에 남은 `from experiments.configs import CONFIGS`
  검증 커맨드 문자열 수정 — 이력 보존 대상이며 ver3 범위 밖

## 3. 재구성 매핑

| ver2 원본 | ver3 신규 | 변경 내용 |
| --- | --- | --- |
| `scripts/config.py` | `scripts/config.py` | 그대로 (바이트 동일) |
| `scripts/train.py` | `scripts/train.py` | 그대로 (바이트 동일) |
| `scripts/evaluate.py` | `scripts/evaluate.py` | 그대로 (바이트 동일) |
| `scripts/predict.py` | `scripts/predict.py` | 그대로 (바이트 동일) |
| `experiments/configs.py` | `scripts/batch_config.py` | 헤더 주석 경로만 변경, 본문 동일 |
| `experiments/run.py` | `scripts/batch_run.py` | 헤더 주석 경로 변경 + import 1줄 변경 |
| `experiments/__init__.py` | (없음) | `experiments/` 폴더 제거로 불필요 |

`scripts/`는 그대로 두고, `experiments/`의 두 파일만 파일명을 바꿔 같은 `scripts/`에 병합하는
구조다. 병합 후 ver3 `scripts/` 폴더는 다음과 같다.

```text
scripts/
├── batch_config.py     # experiments/configs.py 리네임
├── batch_run.py        # experiments/run.py 리네임 + import 수정
├── config.py           # ver2 원본 그대로
├── evaluate.py         # ver2 원본 그대로
├── predict.py          # ver2 원본 그대로
└── train.py            # ver2 원본 그대로
```

이름 충돌 검토: ver2 `scripts/`에는 `config.py`(단수), `train.py`, `evaluate.py`, `predict.py`가
있다. 신규 파일명은 `batch_config.py`, `batch_run.py`로 기존 어느 파일과도 충돌하지 않는다.

경로 계약 유지:
- `batch_run.py`의 `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`은
  `experiments/`에서든 `scripts/`에서든 동일하게 프로젝트 루트(한 단계 위)를 가리키므로 그대로 유효하다.
- `run(mode, ...)`의 `script = os.path.join("scripts", "%s.py" % mode)`도 대상 경로가 그대로이므로
  변경 불필요하다.
- 이관한 `config.py`/`train.py`/`evaluate.py`/`predict.py`는 모두 `scripts.config` 및 `src.*`를
  import 한다. `src/`는 이번 범위에 없으므로 실제 실행은 후속 plan에서 `src/` 이관 후 가능해진다.

## 4. 완료 기준

- ver3 `scripts/`에 `config.py`, `train.py`, `evaluate.py`, `predict.py`가 ver2와 바이트 동일하게
  존재할 것
- ver3 `scripts/`에 `batch_config.py`, `batch_run.py`가 생성될 것
- `batch_config.py` 본문이 ver2 `experiments/configs.py`와 동일하고 헤더만 새 경로일 것
- `batch_run.py`가 `from scripts.batch_config import CONFIGS`를 import 하고, 그 외 로직은 ver2
  `experiments/run.py`와 동일할 것
- ver2 파일은 하나도 수정되지 않을 것

## 5. 검증

ver3에는 `src/` 등 의존 모듈이 없으므로 실제 실행(import) 검증은 이 단계에서 불가능하다. 이관은
텍스트 동등성으로 검증한다.

- `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`가 각각
  ver2 원본과 `diff` 결과 무차이임을 확인
- `scripts/batch_config.py`와 ver2 `experiments/configs.py`를 비교해 헤더 1줄 외 차이가 없음을 확인
- `scripts/batch_run.py`와 ver2 `experiments/run.py`를 비교해 헤더 1줄과 import 1줄 외 차이가 없음을 확인
