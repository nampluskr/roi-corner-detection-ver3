# Use Cases

이 문서는 CLI에서 자주 수행하는 네 가지 시나리오를 실행 절차로 설명한다. 각 시나리오는 사용할 config
파일과 command, 산출물, 확인 방법을 함께 다룬다. option의 전체 목록과 의미는 [CLI Usage](02-cli-usage.md)에,
산출물 파일 구조는 [Experiment Output](04-experiment-output.md)에, 반복 실험 절차는
[Training Workflow](03-training-workflow.md)에 있다.

네 시나리오는 다음과 같다.

| 시나리오 | 목적 | 주요 command |
| --- | --- | --- |
| 1 | public dataset 학습과 평가 집계 | `batch_run.py --config configs/public.py`, `collect_metrics.py` |
| 2 | synthetic dataset 학습과 평가 집계 | `batch_run.py --config configs/synthetic.py`, `collect_metrics.py` |
| 3 | measured dataset 학습과 평가 집계 | `batch_run.py --config configs/measured.py`, `collect_metrics.py` |
| 4 | measured dataset 추론만 수행 | `batch_run.py --mode predict --config configs/predict.py` 또는 `predict.py` |

## 1. 사전 준비

모든 시나리오는 conda 환경 `pytorch_env`와 project root에서 실행한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

각 시나리오의 실험 조합은 `configs/` 아래 stage별 파일에서 정의한다. 파일 구성은 다음과 같다.

| 파일 | dataset stage | 용도 |
| --- | --- | --- |
| `configs/public.py` | `public` | public 학습과 평가 대상 config |
| `configs/synthetic.py` | `synthetic` | synthetic 학습과 평가 대상 config |
| `configs/measured.py` | `measured` | measured 학습과 평가 대상 config |
| `configs/predict.py` | `measured` | metric 평가 없이 추론만 수행할 config |

각 config 파일은 batch runner가 요구하는 `CONFIGS` 이름을 정의한다. `CONFIGS`는 dict의 list이며 각 dict는
`model`, `network`, `head` 같은 assembly option과 함께 `dataset`, `csv_path`를 포함한다. 실제 실행 조합은
사용자가 파일에서 직접 채운다.

## 2. 시나리오 1: public dataset 학습과 평가 집계

public dataset에서 여러 model, network, head를 학습하고 평가 결과를 하나의 CSV로 모은다. 먼저
`configs/public.py`의 `CONFIGS`에 실행할 조합을 채운 뒤 다음을 실행한다.

```bash
python scripts/batch_run.py --mode all --config configs/public.py
```

`--mode all`은 config 전체에 대해 train, evaluate, predict를 순서대로 실행한다. 학습만 하려면
`--mode train`, 평가만 하려면 `--mode evaluate`를 사용한다. runner는 config별로 output directory에
`history.json`, `model.pth`, `metrics.json`, `predictions.csv`를 남긴다.

평가가 끝나면 여러 experiment의 `metrics.json`을 한 표로 모은다.

```bash
python scripts/collect_metrics.py --dataset public --output outputs/public/metrics_summary.csv
```

`collect_metrics.py`는 `outputs/public/` 아래 experiment directory를 순회하며 `metrics.json`을 pandas
DataFrame으로 읽어 `dataset`, `model`, `network`, `head`, `exp_name`과 metric 열을 가진 CSV로 저장한다.
`--dataset`을 생략하면 `outputs/` 전체를 집계한다.

## 3. 시나리오 2: synthetic dataset 학습과 평가 집계

synthetic dataset에서 같은 절차를 수행한다. `configs/synthetic.py`의 `CONFIGS`를 채운 뒤 실행한다.

```bash
python scripts/batch_run.py --mode all --config configs/synthetic.py
python scripts/collect_metrics.py --dataset synthetic --output outputs/synthetic/metrics_summary.csv
```

synthetic config의 dict가 public 단계와 같은 `model`, `network`, `head`, experiment 이름을 사용하면
`train.py`가 `public` 단계의 `model.pth`에서 weight를 자동으로 이어받는다. 이 staged 초기화는
[Data Strategy](../architecture/05-data-strategy.md)에서 다룬다. 이전 단계 checkpoint가 없으면 error 없이
backbone init에서 학습한다.

## 4. 시나리오 3: measured dataset 학습과 평가 집계

measured dataset에서 같은 절차를 수행한다. `configs/measured.py`의 `CONFIGS`를 채운 뒤 실행한다.

```bash
python scripts/batch_run.py --mode all --config configs/measured.py
python scripts/collect_metrics.py --dataset measured --output outputs/measured/metrics_summary.csv
```

measured config가 synthetic 단계와 같은 assembly와 experiment 이름을 사용하면 `train.py`가 `synthetic`
단계의 `model.pth`에서 weight를 이어받는다. 세 stage를 순서대로 실행하면 public, synthetic, measured로
weight가 carryover된다.

## 5. 시나리오 4: measured dataset 추론만 수행

measured dataset에서 특정 model, network, head를 선택해 metric 평가 없이 바로 추론한다. 추론은 대상
image에 대한 corner 예측을 `predictions.csv`로 저장하며 metric을 계산하지 않는다. 두 가지 방법이 있다.

여러 조합을 추론할 때는 `configs/predict.py`의 `CONFIGS`에 대상 조합을 채우고 `--mode predict`로 실행한다.

```bash
python scripts/batch_run.py --mode predict --config configs/predict.py
```

단일 조합만 추론할 때는 config 파일 없이 `predict.py`를 직접 실행한다. `--checkpoint`를 생략하면 현재
stage output 경로의 `model.pth`를 사용한다.

```bash
python scripts/predict.py --dataset measured --model reg --network custom --head gap \
--csv_path /absolute/path/to/gt_corners.csv --checkpoint outputs/measured/reg/custom_gap/EXP/model.pth
```

두 방법 모두 evaluate를 실행하지 않으므로 `metrics.json`을 만들지 않는다. 예측 결과 열 구조는
[Experiment Output](04-experiment-output.md)의 `predictions.csv` 절에서 확인한다.

## 6. 확인 방법

각 시나리오의 정상 동작은 다음으로 확인한다.

- batch run terminal에서 config별 `[OK]`와 최종 summary의 success count를 확인한다. 중간에 `[FAIL]`이
  있으면 해당 config의 output과 오류 메시지를 확인한다.
- 학습 output directory에 `history.json`과 `--save` 시 `model.pth`가 생성됐는지 확인한다.
- 평가 output directory에 `metrics.json`이 생성됐는지 확인하고, 집계 CSV의 행 수가 실행한 experiment 수와
  맞는지 확인한다.
- 추론 시나리오에서는 `predictions.csv`가 생성되고 `metrics.json`이 없는지 확인한다.

## 7. 흔한 오류

시나리오 실행 중 자주 만나는 오류와 조치는 다음과 같다.

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| `batch config must define CONFIGS` | config 파일에 `CONFIGS`가 없음 | 파일에 top-level `CONFIGS` list 정의 |
| evaluate와 predict에서 `FileNotFoundError` | stage `model.pth`가 아직 없음 | 먼저 `--mode train` 또는 `--save`로 checkpoint 생성 |
| 모든 실험이 `public`으로 실행됨 | config dict에 `dataset` 누락 | 각 dict에 `dataset` 명시 |
| 집계 CSV가 비어 있음 | 대상 `outputs/<dataset>/`에 `metrics.json` 없음 | 먼저 evaluate 실행 후 집계 |

## 8. 핵심 요약

stage별 학습과 평가 집계는 `configs/`의 dataset 파일과 `batch_run.py`, `collect_metrics.py`를 조합해
수행한다. 시나리오 1-3은 `--mode all`로 학습과 평가, 예측을 실행한 뒤 `collect_metrics.py`로 여러 실험의
metric을 하나의 CSV로 모은다. 시나리오 4는 `--mode predict` 또는 `predict.py`로 metric 평가 없이 추론만
수행한다. 각 config dict에 `dataset`과 `csv_path`를 명시해야 stage와 data source가 올바르게 전달된다.
