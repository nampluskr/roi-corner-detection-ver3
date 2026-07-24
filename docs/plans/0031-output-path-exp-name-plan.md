# Output Path와 Exp Name 변경 계획

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | output path 생성 로직, metrics 집계, canonical 문서 |
| 관련 문서 | `README.md`, `docs/README.md`, `docs/guides/04-experiment-output.md`, `docs/guides/02-cli-usage.md`, `docs/architecture/04-training-and-inference-flow.md`, `docs/architecture/02-model-assembly.md`, `docs/architecture/05-data-strategy.md`, `docs/glossary.md` |

## 목적과 배경

기존 자동 output directory는 `network_head` segment와 `exp_name` 안에 network와 head 정보를 중복해서
담았다. 실험 산출물 경로를 더 짧고 직접적으로 만들기 위해 자동 경로를
`outputs/<dataset>/<model>/<exp_name>/`로 단순화하고, experiment name에는 model, network, head, dataset을
담는다.

`exp_name`과 output path에는 `batch_size`, `max_epochs`를 포함하지 않는다. 두 값은 stage 학습 시
dataset 규모에 따라 달라질 수 있는 hyperparameter이고, `get_prev_checkpoint_path()`는 이전 dataset stage의
checkpoint를 같은 `cfg` 값으로 조회한다. `exp_name`에 이 값들이 들어가면 stage 간 `batch_size`,
`max_epochs`가 다를 때 이전 단계 checkpoint 경로를 찾지 못하는 문제가 생긴다.

## 범위

포함 항목은 다음과 같다.

- `scripts/config.py`의 experiment name 함수명과 output path 생성 규칙을 변경한다.
- train, evaluate, predict, batch 실행에서 새 output path를 사용한다.
- metrics 집계가 새 3단계 experiment directory를 파싱하게 한다.
- 현재 상태를 설명하는 canonical 문서와 guide, architecture 문서를 새 규칙으로 갱신한다.

제외 항목은 다음과 같다.

- 이미 생성된 `outputs/` 산출물 directory의 migration은 수행하지 않는다.
- 완료된 과거 plan 문서는 이력으로 유지하고 수정하지 않는다.

## 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `get_exp_name()`이 `<model>_<network>_<head>_<dataset>` 형식의 이름을 반환한다.
- `get_output_dir()`가 `outputs/<dataset>/<model>/<exp_name>/` 구조를 반환한다.
- `get_exp_name()`과 `get_output_dir()`가 `batch_size`, `max_epochs`를 exp_name에 포함하지 않아
  `get_prev_checkpoint_path()`가 stage 간 `batch_size`, `max_epochs` 차이와 무관하게 이전 단계
  checkpoint 경로를 찾는다.
- `collect_metrics.py`가 새 경로에서 `dataset`, `model`, `exp_name`과 metric 열을 가진 CSV를 생성한다.
- 현재 문서의 output path 설명과 예시가 새 규칙과 일치한다.

## 검증

검증은 `pytorch_env`에서 수행한다.

- `get_exp_name`, `get_output_dir`, `get_checkpoint_path`, `get_prev_checkpoint_path` 반환값을 import 수준에서
  확인한다.
- `batch_size`, `max_epochs`가 다른 두 `cfg`로 `get_prev_checkpoint_path`를 호출해 동일한 경로를 반환하는지
  확인한다.
- `/tmp`의 임시 metrics tree로 `collect_metrics.py` 집계 결과 열을 확인한다.
- `rtk rg`로 current code와 canonical 문서에 old canonical path와 `get_experiment()` 호출이 남지 않았는지
  확인한다.
