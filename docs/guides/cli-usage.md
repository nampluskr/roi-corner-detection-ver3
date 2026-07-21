# CLI Usage

학습, 평가, 예측은 각각 `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`로 실행한다. 세
script는 공통 parser를 사용하므로 model 조립과 data split 인자가 일관되게 적용된다.

## Common Arguments

주요 인자는 다음과 같다.

| 인자 | 기본값 | 의미 |
| --- | --- | --- |
| `--dataset` | `public` | output 경로의 논리 data stage |
| `--model` | `reg` | model package 선택자 |
| `--network`, `--net` | `custom` | backbone 또는 external whole-model 이름 |
| `--head` | `gap` | model별 head 또는 detection target 표현 |
| `--image_size` | `224` | resize될 정사각 image edge |
| `--batch_size` | `4` | batch size |
| `--output_dir` | 없음 | 기본 output 경로를 대신할 명시 경로 |
| `--device` | 자동 | PyTorch device 문자열 |

`--csv_path`는 하나 이상의 CSV 경로를 받을 수 있다. 지정하지 않으면 configuration의 기본 public CSV
목록을 사용한다. `--train_size`, `--valid_size`, `--test_size`는 각 split에서 사용할 표본 수를 제한한다.

## Training

다음은 custom coordinate regression 학습 예시다.

```bash
python scripts/train.py --model reg --network custom --head gap --save
```

dense segmentation과 external detection은 같은 형태로 실행한다.

```bash
python scripts/train.py --model seg --network custom --head mask --save
python scripts/train.py --model yolo --network yolov8n --head box --save
```

`--max_epochs`, `--patience`, `--warmup_epochs`로 training lifecycle을 조정한다. checkpoint 저장은
`--save`를 지정한 경우에만 수행한다.

## Evaluation and Prediction

평가와 예측은 같은 model assembly option을 checkpoint 생성 시점과 맞춰야 한다.

```bash
python scripts/evaluate.py --model reg --network custom --head gap --checkpoint outputs/.../model.pth
python scripts/predict.py --model reg --network custom --head gap --checkpoint outputs/.../model.pth
```

evaluation은 `metrics.json`, prediction은 `predictions.csv`를 output directory에 기록한다. 서로 다른
assembly option으로 wrapper를 만들면 checkpoint state가 일치하지 않을 수 있다.
