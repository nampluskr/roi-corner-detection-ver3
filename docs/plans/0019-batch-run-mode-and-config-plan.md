---
상태: Done
작성일: 2026-07-22
적용 범위: `scripts/batch_run.py` CLI에 `--mode all`과 `--config` 설정 파일 선택 기능 추가, batch 실행 guide 갱신
관련 문서: [../guides/02-cli-usage.md](../guides/02-cli-usage.md), [../guides/03-training-workflow.md](../guides/03-training-workflow.md), [../../scripts/batch_run.py](../../scripts/batch_run.py), [../../scripts/batch_config.py](../../scripts/batch_config.py)
---

## 목적과 배경

현재 `scripts/batch_run.py`는 `train`, `evaluate`, `predict` 중 한 mode만 선택할 수 있고
`scripts.batch_config.CONFIGS`를 고정 import한다. 따라서 학습 후 평가와 예측을 완료하려면 세
command를 별도로 실행해야 하며, 실험 묶음별 설정을 분리하려면 기본
`scripts/batch_config.py`를 계속 수정해야 한다.

이 작업은 `--mode all`로 `train`, `evaluate`, `predict`를 순서대로 실행하고,
`--config xxx_config.py`로 `CONFIGS`를 정의한 별도 Python 파일을 선택할 수 있게 한다.
기존 option을 사용하지 않는 command의 동작은 유지한다.

## 범위

포함 항목은 다음과 같다.

- `scripts/batch_run.py`의 `--mode` 선택지에 `all`을 추가한다. `all`은 전체
  config에 대한 `train`을 완료한 뒤 `evaluate`, `predict` 순서로 실행한다.
- `--config` option을 추가해 project root 또는 현재 작업 directory 기준의 Python 파일
  path를 받는다. 해당 파일은 top-level `CONFIGS`를 정의해야 한다.
- `--config`를 생략하면 기존 `scripts/batch_config.py`를 사용하여 하위 호환성을
  유지한다.
- 설정 파일이 없거나 load할 수 없는 경우, `CONFIGS`가 없는 경우에 실행 전
  명확한 error를 보고한다.
- `all` 실행 중 개별 config가 실패해도 기존 방식처럼 나머지 config와 후속
  mode를 계속 실행하고, 최종적으로 하나라도 실패하면 exit code 1을 반환한다.
- `docs/guides/02-cli-usage.md`와 `docs/guides/03-training-workflow.md`의 batch command,
  설정 파일 규약, `all` 순서와 실패 처리 설명을 갱신한다.

제외 항목은 다음과 같다.

- batch config dictionary에서 전달할 수 있는 `PASS_KEYS`는 확장하지 않는다.
- config schema를 Python 외의 JSON, YAML 형식으로 확장하지 않는다.
- subprocess 병렬 실행, 실패 config 재시도, checkpoint resume는 추가하지 않는다.
- 기존 `scripts/batch_config.py`의 활성 config 내용은 변경하지 않는다.

## 완료 기준

다음을 모두 충족하면 이 plan을 `Done`으로 본다.

- `python scripts/batch_run.py --mode all`이 기본 `CONFIGS`에 대해 `train`, `evaluate`,
  `predict` 순서를 사용한다.
- `python scripts/batch_run.py --mode train --config path/to/xxx_config.py`가 지정한
  파일의 `CONFIGS`만 사용한다.
- `--config`를 생략한 기존 train, evaluate, predict command가 동일하게 동작한다.
- 잘못된 config path와 `CONFIGS`가 없는 파일이 subprocess 실행 전에 실패한다.
- guide 문서가 새 option과 실제 command 예시를 반영한다.

## 검증

다음 항목으로 검증한다.

- conda 환경 `pytorch_env`에서 `python -m py_compile scripts/batch_run.py` 명령으로
  syntax를 확인한다.
- 임시 config 파일과 subprocess mock을 사용해 기본 config load, 외부 config load,
  `all` mode의 순서, 실패 exit code를 검증한다. 실제 장시간 학습은 수행하지
  않는다.
- `--help`에 `train`, `evaluate`, `predict`, `all`과 `--config` 설명이 표시되는지
  확인한다.
- 갱신한 Markdown가 project 문서 규칙을 준수하는지 확인한다.

검증 결과 `py_compile`과 `--help`가 통과했다. subprocess mock으로 `all` mode가
`train`, `evaluate`, `predict` 순서를 모두 호출하고 중간 실패를 최종 exit code 1로
반영하는 것을 확인했다. 기본 설정, 절대 path의 외부 설정, project root 기준 상대
path 설정을 load했고, 없는 path와 `CONFIGS`가 없는 Python 파일의 error를 확인했다.
