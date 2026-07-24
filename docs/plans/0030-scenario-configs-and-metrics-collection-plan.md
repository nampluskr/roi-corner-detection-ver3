# 0030 Scenario Configs and Metrics Collection Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `configs/`, `scripts/collect_metrics.py`, `scripts/batch_config.py`, `docs/guides/06-use-cases.md`, `docs/guides/03-training-workflow.md`, `docs/README.md` |
| 관련 문서 | [02-cli-usage.md](../guides/02-cli-usage.md), [03-training-workflow.md](../guides/03-training-workflow.md), [04-experiment-output.md](../guides/04-experiment-output.md), [0019-batch-run-mode-and-config-plan.md](0019-batch-run-mode-and-config-plan.md), [0029-staged-dataset-training-plan.md](0029-staged-dataset-training-plan.md) |

## 1. 목적과 배경

현재 batch 실행은 `scripts/batch_config.py` 하나에 모든 model 계열 config를 주석 template로 모아 두고
`CONFIGS`로 합쳐 실행한다. 이 구조는 dataset stage별로 어떤 실험을 돌릴지 구분하기 어렵고, `dataset`과
`csv_path`를 config마다 반복해서 넣어야 한다. 또한 여러 실험의 평가 결과가 각 experiment directory의
`metrics.json`으로만 흩어져 있어, 한 stage의 model, network, head 조합을 한눈에 비교하는 표가 없다.

사용자는 CLI에서 다음 네 시나리오를 실행하려고 한다.

1. `public` dataset에서 여러 model, network, head를 학습하고 평가 결과를 하나의 CSV로 모은다.
2. `synthetic` dataset에서 같은 절차를 수행한다.
3. `measured` dataset에서 같은 절차를 수행한다.
4. `measured` dataset에서 특정 model, network, head를 선택해 metric 평가 없이 바로 추론한다.

이 plan은 stage별 config 파일 분리와 metric 집계 script를 추가하고, 네 시나리오를 CLI 절차로 설명하는
use case 문서를 만든다.

## 2. 범위

포함하는 작업은 다음과 같다.

- `configs/` 폴더를 만들고 stage별 config 파일을 둔다. 각 파일은 batch_run이 요구하는 `CONFIGS` 이름을
  정의하고, dict마다 `dataset`과 `csv_path`를 포함한다. 파일 구성은 다음과 같다.
  - `configs/public.py`: `dataset` = `public` 실험 config
  - `configs/synthetic.py`: `dataset` = `synthetic` 실험 config
  - `configs/measured.py`: `dataset` = `measured` 실험 config
  - `configs/predict.py`: `dataset` = `measured`에서 추론만 수행할 선택 config
- 각 config 파일의 실제 항목은 사용자가 채울 수 있도록 model, network, head 조합 예시를 주석 template로
  제공하고, 활성 `CONFIGS`는 최소 한 개의 동작 가능한 예시를 둔다. `predict.py` config는 metric 평가가
  필요 없는 추론 대상만 나열한다.
- `scripts/collect_metrics.py`를 추가한다. 이 script는 `outputs/<dataset>/` 아래 experiment directory를
  순회하며 `metrics.json`을 읽어 pandas DataFrame으로 모으고, `dataset`, `model`, `network`, `head`,
  `exp_name`과 metric 열을 가진 CSV로 저장한다. `--dataset`으로 특정 stage만 집계하고, 생략하면 `outputs/`
  전체를 집계한다. 출력 경로는 `--output`으로 지정하며 기본값을 정의한다.
- `scripts/batch_config.py`는 기존 default config로 유지하되, `configs/`로 분리된 stage config와의 관계를
  파일 상단 주석으로 안내한다. 기존 template 항목 값은 바꾸지 않는다.
- `docs/guides/06-use-cases.md`를 추가하여 네 시나리오를 CLI 실행 절차로 설명한다. 각 시나리오는 사용할
  config 파일, `batch_run.py` 명령, 산출물, 확인 방법을 포함한다. 시나리오 1-3은 `collect_metrics.py`로
  CSV를 만드는 단계를, 시나리오 4는 `predict.py` 또는 `batch_run.py --mode predict`로 추론만 수행하는
  단계를 설명한다.
- `docs/README.md`의 guide 색인과 문서 구조 tree에 `06-use-cases.md`를 추가한다.
- `docs/guides/03-training-workflow.md`의 batch run 절에서 stage별 config 분리와 metric 집계 script를
  참조하도록 갱신한다.

제외하는 작업은 다음과 같다.

- 세 stage를 하나의 명령으로 자동 orchestrate하는 pipeline은 만들지 않는다. staged weight carryover는
  0029에서 구현한 `train.py`의 dataset 기반 자동 초기화를 그대로 사용한다.
- 단계별 learning rate 스케줄 자동 조정은 범위에 포함하지 않는다.
- `experiments/` 폴더는 만들지 않는다. 실행 진입점은 기존 `scripts/`와 새 `configs/`를 사용한다.
- `src/` 아래 학습, 평가, 추론 코드의 동작은 바꾸지 않는다. `collect_metrics.py`는 산출물만 읽는다.
- config 파일의 실제 대량 실험 조합을 채우는 작업은 사용자 몫이며, 이 plan은 동작 가능한 최소 예시만
  제공한다.

## 3. 완료 기준

다음을 모두 충족하면 이 plan을 `Done`으로 본다.

- `configs/public.py`, `configs/synthetic.py`, `configs/measured.py`, `configs/predict.py`가 존재하고,
  각각 `CONFIGS`를 정의하며 dict마다 `dataset`과 `csv_path`를 포함한다.
- `scripts/collect_metrics.py`가 존재하고, `outputs/` 아래 `metrics.json`을 pandas DataFrame으로 모아
  CSV로 저장하며, `--dataset` 필터와 `--output` 경로를 지원한다.
- `python scripts/batch_run.py --mode all --config configs/public.py` 형태로 네 시나리오를 실행하는
  명령이 문서에 명시되고, 시나리오 4는 `--mode predict`로 추론만 수행한다.
- `docs/guides/06-use-cases.md`가 네 시나리오를 CLI 절차로 설명하고, `docs/README.md` 색인과 구조 tree에
  반영된다.
- 새 Python 파일은 project code 규칙(첫 줄 header, os.path, no type hint, ASCII 화살표 등)을 따른다.

## 4. 검증

다음으로 검증한다.

- `configs/`의 네 파일을 `python -c`로 import하여 `CONFIGS`가 list이고 각 dict에 `dataset`과 `csv_path`가
  있는지 확인한다.
- `scripts/collect_metrics.py`를 `--help`로 실행해 인자 파싱이 동작하는지 확인하고, 임시로 만든 소수의
  `metrics.json`에 대해 CSV 집계가 올바른 열을 만드는지 확인한다.
- `scripts/batch_run.py`가 `--config configs/public.py`로 config를 load하는지 확인한다. 실제 학습은
  data와 시간이 필요하므로 dry한 config load와 `get_cli_args` 결과 확인까지 수행한다.
- 문서 생성 항목은 새 문서 존재, 색인 반영, 화살표와 code block 규칙 준수를 확인한다.
