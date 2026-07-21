@/home/nampl/.codex/RTK.md

# 프로젝트 에이전트 지침

## 1. Canonical project design

이 프로젝트는 `260712_roi-corner-detection-ver2`의 폴더와 파일 구성을 재구성한 결과를 확립하는
workspace다. ver2의 SSOT/canonical 문서는 참고 자료로만 사용하며, ver3의 구속력 있는 기준은 ver3
내 문서와 구현 결과로 확립한다.

현재 ver3의 문서 기준은 다음과 같다.

| 문서 | 상태 | 역할 |
| --- | --- | --- |
| `README.md` | canonical | 프로젝트 목적과 재구성 대상 |
| `docs/README.md` | canonical | 문서 색인과 작업 절차 |
| `docs/plans/*.md` | history | 작업 계획과 완료 이력 |

ver2 참고 경로는 다음과 같다.

- 프로젝트: `../260712_roi-corner-detection-ver2/`
- 기존 SSOT: `../260712_roi-corner-detection-ver2/docs/architecture/model-assembly.md`

ver2 내부 파일은 참고와 읽기 전용으로만 사용하며 직접 수정하지 않는다.

## 2. 작업 범위와 산출물

구현과 문서는 현재 workspace에 존재하는 범위에서 작업한다. 새 data, src, experiments 또는 outputs
folder는 사용자 요청 또는 ver3 plan의 구현 단계가 없으면 만들지 않는다.

실험 산출물 경로는 ver2 규칙을 ver3에도 적용한다.

```text
outputs/<dataset>/<method>/<model>/<exp_name>/
```

`dataset`은 `public`, `synthetic`, `measured`의 논리 stage다. method, model과 variant의 의미는
ver3에서 확정된 plan과 구현 결과를 따른다.

## 3. 문서 작성 규칙

모든 Markdown 문서는 다음 규칙을 따른다.

- 본문은 서술체를 사용한다.
- em dash, 유니코드 화살표, 이모지를 사용하지 않는다.
- Markdown 본문의 화살표는 `$\to$`를 사용한다.
- fenced code block과 inline code 안에서는 ASCII `->`를 사용한다.
- 폴더 구조 tree는 `├ ─ │ └` 문자를 사용한다.
- header level을 건너뛰지 않고 H4 아래 level은 사용하지 않는다.
- 수평 구분선은 사용하지 않는다. YAML frontmatter의 `---`는 예외다.
- table과 list 앞에는 내용을 소개하는 문장을 둔다.
- 폴더와 파일 목록은 폴더를 알파벳순으로 먼저 나열하고 파일을 알파벳순으로 나열한다.
- Jupyter notebook cell의 `source` 배열 마지막 원소는 줄바꿈으로 끝나지 않는다.

## 4. 코드 작성 규칙

모든 Python 코드는 다음 규칙을 따른다.

- 식별자, 주석, docstring, 문자열에 한국어를 사용하지 않는다.
- 세로 정렬을 위한 불필요한 공백을 넣지 않는다.
- 경로 처리는 `pathlib.Path` 대신 `os.path`를 사용한다.
- type hint를 사용하지 않는다.
- 모든 파일의 첫 줄은 `# path/from/project/root.py: one-line description` 형식으로 작성한다.
- 첫 줄 header 다음에 빈 줄 하나를 두고 import를 작성한다.
- class와 top-level function은 한 줄 docstring을 작성한다.
- method에는 docstring을 작성하지 않는다.
- 주석은 필요한 경우에만 최소한으로 작성한다.
- `src/` 아래 모든 폴더에는 빈 `__init__.py`를 둔다.
- `src/` 내부 import는 `src.xxx` 형식의 absolute import를 사용한다.
- `scripts/`에서는 project root를 `sys.path`에 추가한 뒤 `scripts.xxx`와 `src.xxx`로 import한다.
- 향후 `experiments/`를 만들면 project root를 `sys.path`에 추가한 뒤 `src.xxx`로 import한다.

`scripts/` 바로 아래 Python 파일은 다음 project root 추가 패턴을 사용한다.

```python
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

현재 ver3 project root의 절대 경로는 다음과 같다.

```text
/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

## 5. 실행 환경과 Python 경로

Python 실행과 검증은 conda 환경 `pytorch_env`를 사용한다. 코드 실행, `python -c` 검증,
스크립트 실행 전에 먼저 이 환경을 활성화하고 ver3 project root에서 실행한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

일회성 import 검증처럼 script의 `sys.path` 보정이 적용되지 않는 명령에서는 `PYTHONPATH`에 ver3
project root를 포함한다.

```bash
PYTHONPATH=/mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3 python -c "import src"
```

## 6. Plan 문서 규칙

Canonical 문서를 갱신할 정도의 실제 작업, 구현, 구조 변경, 문서 체계 확장 등은 실행 전에
`docs/plans/`의 계획 문서에 범위와 완료 기준을 기록하고 검토와 승인을 받는다. 요구사항이나 설계가
바뀌면 코드보다 canonical 문서를 먼저 수정한다.

계획 문서의 경로와 명명 규칙은 다음과 같다.

- 경로: `docs/plans/NNNN-topic-plan.md`
- 번호 `NNNN`은 4자리 0-padding이며 순증가한다. 번호를 재사용하거나 삭제하지 않는다.
- 상태는 `Draft`, `Approved`, `Done` 중 하나를 사용한다.
- 완료된 plan도 파일을 지우지 않고 이력으로 보존한다.

각 plan 문서는 다음 구성 요소를 갖춘다.

- 표준 헤더 표: 상태, 작성일, 적용 범위, 관련 문서
- 목적과 배경: 이 작업이 왜 필요한지
- 범위: 포함 항목과 제외 항목을 구분해서 기록
- 완료 기준: 무엇이 충족되면 이 plan을 `Done`으로 볼 수 있는지
- 검증: 빌드 또는 테스트 방법, 문서 생성만이면 그 사실과 확인 항목

작업 순서는 다음과 같다.

1. 작업 전에 관련 plan이 있는지 `docs/plans/`를 확인한다.
2. 없으면 새 `NNNN-topic-plan.md` 초안을 작성해 사용자에게 검토와 승인을 받는다.
3. 승인된 plan을 기준으로 canonical 문서를 먼저 갱신하고, 이후 코드를 구현한다.
4. 구현 결과에 맞게 관련 검증을 수행하고 plan의 상태를 갱신한다.

## 7. 동기화 규칙

`CLAUDE.md`와 `AGENTS.md`는 같은 작업 지침의 동기화 사본이다. 한 파일의 내용이 변경되면 같은
작업에서 다른 파일을 동일한 내용으로 갱신하고 SHA-256으로 byte-level 일치를 검증한다.
