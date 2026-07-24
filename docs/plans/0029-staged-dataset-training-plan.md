# 0029 Staged Dataset Training Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `scripts/config.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `README.md`, `docs/guides/02-cli-usage.md`, `docs/guides/04-experiment-output.md`, `docs/architecture/05-data-strategy.md` |
| 관련 문서 | [05-data-strategy.md](../architecture/05-data-strategy.md), [02-cli-usage.md](../guides/02-cli-usage.md), [04-experiment-output.md](../guides/04-experiment-output.md) |

## 1. 목적과 배경

이 프로젝트는 `public`, `synthetic`, `measured` 세 논리 dataset stage를 transfer-learning 전략으로
정의하며, 그 흐름은 `docs/architecture/05-data-strategy.md`에 기술되어 있다. 현재 `--dataset` 인자는
출력 경로 segment를 결정하는 label 역할만 하고 가중치 로딩에는 관여하지 않는다. 그래서
`scripts/train.py`는 항상 backbone init 상태에서 학습을 시작하고, `scripts/evaluate.py`와
`scripts/predict.py`는 명시적 `--checkpoint`를 요구한다.

이 작업의 목적은 stage 단계별 fine-tuning을 실제 실행 흐름에 연결하는 것이다. later stage를 학습할
때 이전 stage에서 저장한 가중치로 초기화한다. 구체적으로 `--dataset public`은 backbone init에서
학습하고 (기존과 동일), `--dataset synthetic`은 `outputs/public` 아래 저장된 `model.pth`를 불러와
이어서 학습하며, `--dataset measured`는 `outputs/synthetic` 아래 저장된 `model.pth`를 불러와 이어서
학습한다.

사용자와 확정한 설계 결정은 다음과 같다.

- 이전 stage 가중치 경로는 auto-derive 방식으로만 해석한다. 현재 config에서 파생한 경로의 `dataset`
  segment만 이전 stage로 바꿔 재사용하며 별도 `--init_from` 인자는 추가하지 않는다. 이 방식은 이전
  stage가 동일한 `model`, `network`, `head`, `batch_size`, `max_epochs`로 학습되어 파생 `exp_name`이
  일치한다고 가정한다.
- 이전 stage `model.pth`가 없으면 error가 아니라 backbone init으로 fallback하여 학습하고 로그를
  남긴다.
- `--dataset` 기반 경로 로직을 세 script 모두에 적용한다. `train.py`는 이전 stage를 init으로
  자동 로딩하고, `evaluate.py`와 `predict.py`는 `--checkpoint`가 없으면 현재 stage의 `model.pth`를
  auto-derive한다.

## 2. 범위

포함하는 작업은 다음과 같다.

- `scripts/config.py`에 stage 매핑과 checkpoint 경로 helper를 추가한다. 모듈 수준 매핑
  `PREV_DATASET = {"synthetic": "public", "measured": "synthetic"}`을 두고, `get_output_dir`에
  선택적 `dataset` override 인자를 더한다 (기존 호출부는 override 없이 호출하므로 하위 호환된다).
  `get_checkpoint_path(cfg, dataset=None)`는 해당 stage 출력 폴더 아래 `model.pth` 경로를 반환하고,
  `get_prev_checkpoint_path(cfg)`는 이전 stage의 `model.pth` 경로를 반환하되 이전 stage가 없으면
  `None`을 반환한다.
- `scripts/train.py`에서 `wrapper`와 `trainer`를 구성한 뒤 `fit_early_stop` 이전에 이전 stage
  가중치가 존재하면 로딩한다. `save_model` 옆에 `load_model`을, config에서
  `get_prev_checkpoint_path`를 import한다. 이전 stage checkpoint가 있으면
  `load_model(wrapper.model, prev_checkpoint)`를 호출하고 초기화 로그를 남긴다. 없으면 backbone init
  fallback 로그를 남긴다. `--dataset public`은 이전 stage가 `None`이므로 동작이 바뀌지 않는다.
- `scripts/evaluate.py`와 `scripts/predict.py`에서 `--checkpoint` 필수 guard를 현재 stage
  `model.pth` auto-derive로 대체한다. 명시적 `--checkpoint`는 계속 override한다. config에서
  `get_checkpoint_path`를 import한다.
- canonical 문서를 새 동작에 맞게 갱신한다. `README.md`의 출력 경로와 CLI 설명, 
  `docs/guides/02-cli-usage.md`의 `--dataset` 설명과 `--checkpoint` 선택 사항 안내,
  `docs/guides/04-experiment-output.md`의 `model.pth` auto-derive 설명,
  `docs/architecture/05-data-strategy.md`의 staged transfer-learning 흐름 구현 상태를 반영한다.

제외하는 작업은 다음과 같다.

- stage별 학습 hyperparameter 값 자체나 dataset split 로직은 바꾸지 않는다.
- `src/` 아래 model, wrapper, trainer, dataset loader의 구조는 바꾸지 않는다.
- `scripts/batch_run.py`는 이미 eval/predict에서 `<output_dir>/model.pth`를 auto-derive하므로 바꾸지
  않는다.
- `CLAUDE.md`와 `AGENTS.md`는 변경하지 않으므로 sync 규칙은 적용되지 않는다.

## 3. 완료 기준

- `scripts/config.py`에 `PREV_DATASET`, `get_checkpoint_path`, `get_prev_checkpoint_path`가 있고
  `get_output_dir`가 선택적 `dataset` override를 받는다.
- `scripts/train.py`가 `--dataset synthetic` 또는 `measured`에서 이전 stage `model.pth`를 로딩하고,
  없으면 backbone init fallback 로그를 남긴다. `--dataset public`은 기존 동작을 유지한다.
- `scripts/evaluate.py`와 `scripts/predict.py`가 `--checkpoint` 없이 현재 stage `model.pth`를
  auto-derive하고, 명시적 `--checkpoint`는 계속 우선한다.
- 관련 canonical 문서 네 개가 새 동작을 반영해 갱신되어 있다.

## 4. 검증

conda 환경 `pytorch_env`에서 project root 기준으로 실행한다.

- import 확인:
  ```
  PYTHONPATH=<project-root> python -c "import scripts.config, scripts.train, scripts.evaluate, scripts.predict"
  ```
- `argparse.Namespace`로 만든 가짜 config에 대한 경로 파생 확인:
  - `dataset=public`이면 `get_prev_checkpoint_path`가 `None`을 반환한다.
  - `dataset=synthetic`이면 `outputs/public/<model>/<net_head>/<exp>/model.pth`를 반환한다.
  - `dataset=measured`이면 `outputs/synthetic/<model>/<net_head>/<exp>/model.pth`를 반환한다.
  - override 없는 `get_checkpoint_path`는 현재 stage의 `outputs/<dataset>/.../model.pth`를 반환한다.
- 선택적 소규모 end-to-end 확인:
  - `python scripts/train.py --dataset public --save --train_size 50 --valid_size 20 --max_epochs 1`이
    `outputs/public/.../model.pth`를 생성한다.
  - `python scripts/train.py --dataset synthetic --save --csv_path data/synthetic/gt_corners.csv --train_size 50 --valid_size 20 --max_epochs 1`이 이전 stage 초기화 로그를 남긴다.
  - `python scripts/evaluate.py --dataset synthetic --csv_path data/synthetic/gt_corners.csv`가
    `--checkpoint` 없이 현재 stage `model.pth`를 auto-derive하여 실행된다.
