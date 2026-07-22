# roi-corner-detection-ver3

이 project는 image에서 관심 영역인 ROI의 네 corner를 예측하고 여러 neural network 표현을 같은 기준으로
학습, 평가, 비교하는 PyTorch workspace다. direct coordinate regression, segmentation, dense prediction,
detection, graph refinement, learned mask와 classical geometry의 결합을 하나의 common corner contract로
실행할 수 있다.

처음 사용하는 경우 이 README의 quick start로 data와 실행 환경을 확인한 뒤, [문서 색인](docs/README.md)의
초보자 권장 순서에 따라 architecture와 model 원리를 읽는 것을 권장한다.

## 1. 해결하는 문제

input은 RGB image이고 output은 quadrilateral ROI를 나타내는 네 corner다. common tensor contract는 다음과
같다.

| 항목 | shape | 값과 순서 |
| --- | --- | --- |
| image batch | `(B, 3, H, W)` | ImageNet 방식으로 normalize한 RGB tensor |
| target corner | `(B, 4, 2)` | `[0,1]`, `TL`, `TR`, `BR`, `BL` |
| final prediction | `(B, 4, 2)` | postprocess 이후 normalized ordered corner |

`B`는 batch size다. 기본 input은 `224 x 224`다. coordinate는 pixel index가 아니라 image width와 height에
대한 비율이므로 `(0.25,0.5)`는 image의 왼쪽에서 25%, 위에서 50% 위치를 뜻한다.

model마다 학습 중간 표현은 다르다.

```text
image -> model-specific raw output -> postprocessor -> common final corners
corner target -> preprocessor -> model-specific training target -> loss
```

이 구조 덕분에 mask나 detection box를 학습하는 model도 최종적으로 같은 corner metric으로 비교할 수 있다.

## 2. Project 구조

현재 주요 folder와 file은 다음과 같다. folder를 먼저 알파벳순으로 표시한다.

```text
project-root/
├── docs/
│   ├── architecture/
│   ├── guides/
│   ├── models/
│   ├── plans/
│   ├── reference/
│   ├── glossary.md
│   └── README.md
├── scripts/
│   ├── batch_config.py
│   ├── batch_run.py
│   ├── config.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── src/
│   ├── components/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── __init__.py
└── README.md
```

`scripts/`는 CLI 진입점, `src/data/`는 CSV와 transform, `src/components/`는 shared network 부품,
`src/models/`는 model별 target과 raw output, `src/core/`는 trainer와 evaluator lifecycle을 담당한다.
상세 책임 경계는 [Source Layout](docs/architecture/03-src-layout.md)에서 확인한다.

## 3. 지원 model

CLI의 `--model`이 선택하는 current registry는 다음과 같다.

| model | 학습 표현 | 대표 head | 상세 문서 |
| --- | --- | --- | --- |
| `reg` | normalized corner 8개 직접 회귀 | `gap`, `spatial` | [Regression](docs/models/01-reg.md) |
| `seg` | ROI binary mask | `mask` | [Segmentation](docs/models/02-seg.md) |
| `det` | custom class와 regression grid | `box`, `point` | [Detection](docs/models/04-det.md) |
| `peak` | corner별 Gaussian peak map | `peak` | [Dense Prediction](docs/models/03-dense-prediction.md) |
| `ridge` | edge별 Gaussian ridge map | `ridge` | [Dense Prediction](docs/models/03-dense-prediction.md) |
| `gcn` | graph 기반 iterative corner refinement | `gcn` | [GCN](docs/models/05-gcn.md) |
| `hybrid` | learned mask와 classical geometry | `hybrid` | [Hybrid](docs/models/06-hybrid.md) |
| `torchseg` | torchvision whole segmentation mask | `mask` | [External Models](docs/models/07-external-models.md) |
| `torchdet` | torchvision native detection | `box`, `point` | [External Models](docs/models/07-external-models.md) |
| `yolo` | Ultralytics native detection | `box`, `point` | [External Models](docs/models/07-external-models.md) |
| `detr` | Hugging Face DETR detection | `box`, `point` | [External Models](docs/models/07-external-models.md) |

`model`, `network`, `head`는 서로 다른 선택 축이다. `model`은 target, loss, postprocess package를,
`network` 또는 `net`은 backbone이나 external whole architecture를, `head`는 model별 output variant를 정한다.
가능한 조합의 원리는 [Model Assembly](docs/architecture/02-model-assembly.md)를 참고한다.

## 4. 실행 환경

Python 실행과 검증은 conda environment `pytorch_env`와 project root를 기준으로 한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

external whole-model은 architecture에 따라 torchvision, timm, transformers, ultralytics dependency와 local
pretrained weight가 필요할 수 있다. custom model smoke run은 별도 external weight 없이 공통 pipeline을
확인하기에 적합하다.

## 5. Dataset 준비

labeled CSV의 필수 header는 다음과 같다.

```text
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
```

corner 순서는 `TL`, `TR`, `BR`, `BL`이고 값은 `[0,1]` normalized coordinate여야 한다. `image_dir`과
`image_name`을 결합한 path에 실제 image가 있어야 한다.

current default CSV는 개발 환경의 absolute path이므로 다른 machine에서는 `--csv_path`를 명시해야 한다.
schema, 여러 CSV 결합, 60:20:20 split과 transform은 [Dataset Format Guide](docs/guides/01-dataset-format.md)를
참고한다. data가 public, synthetic, measured 3단계로 구성되는 이유와 단계별 특성이 만드는 project
제약은 [Data Strategy](docs/architecture/05-data-strategy.md)에서 다룬다.

## 6. Quick start training

먼저 작은 custom regression run으로 data loading, forward, loss, validation, checkpoint 저장을 확인한다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --patience 1 --warmup_epochs 0 \
  --num_workers 0 \
  --output_dir outputs/public/reg/custom_gap/quickstart \
  --save
```

정상 종료 후 다음 file을 확인한다.

```text
outputs/public/reg/custom_gap/quickstart/
├── history.json
├── model.pth
└── run.log
```

이 실행은 model 성능 측정이 아니라 integration smoke test다. 높은 IoU보다 loss와 metric이 finite인지,
best weight가 복원되고 checkpoint가 생성됐는지를 확인한다.

## 7. Checkpoint evaluation

같은 CSV, seed, model, network, head와 image size로 test evaluation을 실행한다.

```bash
python scripts/evaluate.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/custom_gap/quickstart/model.pth \
  --output_dir outputs/public/reg/custom_gap/quickstart
```

evaluation은 `metrics.json`에 polygon IoU, MCD, MaxCD, PCK 0.02, PCK 0.05와 success rate를 저장한다.

## 8. Sample prediction

sample별 target과 final corner를 확인하려면 prediction을 실행한다.

```bash
python scripts/predict.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --seed 42 --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/custom_gap/quickstart/model.pth \
  --output_dir outputs/public/reg/custom_gap/quickstart
```

`predictions.csv`에는 target과 prediction의 coordinate 8개, finite success와 failure reason이 기록된다.
success가 true여도 zero 또는 center fallback은 가능하므로 coordinate와 metric을 함께 확인한다.

## 9. CLI 기준

공통 parser의 주요 default는 다음과 같다.

| option | 기본값 | 의미 |
| --- | --- | --- |
| `--dataset` | `public` | output path의 logical data stage |
| `--model` | `reg` | model package 선택자 |
| `--network`, `--net` | `custom` | backbone 또는 whole-model 이름 |
| `--head` | `gap` | model-specific variant |
| `--image_size` | `224` | square resize edge |
| `--batch_size` | `4` | batch sample 수 |
| `--max_epochs` | `5` | 최대 epoch 수 |
| `--patience` | `3` | early stopping wait epoch 수 |
| `--warmup_epochs` | `1` | requested backbone warmup 수 |
| `--num_workers` | `4` | dataloader worker 수 |

global `network=custom`, `head=gap` default는 `reg`에 맞는다. 다른 model을 선택할 때는 compatible head와
network를 명시한다. 전체 option과 예시는 [CLI Usage Guide](docs/guides/02-cli-usage.md)를 참고한다.

## 10. Output path

`--output_dir`을 생략하면 다음 규칙으로 path를 만든다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

`network_head`는 `<network>_<head>`, `exp_name`은
`<model>_bs<batch_size>_ep<max_epochs>_<network>_<head>`다.

automatic name에는 CSV, seed, image size, subset size, patience, warmup이 포함되지 않는다. 이 값만 다른
실험은 같은 file을 덮어쓸 수 있으므로 비교 실험에는 명시적 `--output_dir`을 권장한다. file schema와
metric 해석은 [Experiment Output Guide](docs/guides/04-experiment-output.md)를 참고한다.

## 11. 권장 실험 workflow

한 model 조합은 다음 순서로 검증한다.

1. CSV path, normalized coordinate와 corner order를 확인한다.
2. custom regression one-epoch smoke run으로 공통 pipeline을 확인한다.
3. 원하는 model, network, head 조합을 작은 subset에서 실행한다.
4. `--save` checkpoint로 evaluation과 prediction을 수행한다.
5. history, metrics, predictions를 함께 확인한다.
6. 같은 data identity를 유지하고 sample 수와 epoch를 늘린다.
7. 비교 실험에서는 한 번에 model 또는 network 한 축만 바꾼다.

단계별 checklist는 [Training Workflow Guide](docs/guides/03-training-workflow.md)를 따른다.

## 12. 현재 제약

실행 전에 알아야 할 current limitation은 다음과 같다.

- `--image_size`는 dataloader resize에는 적용되지만 wrapper constructor로 전달되지 않는다. standard
  workflow에서는 224를 유지한다.
- checkpoint는 model state dictionary만 저장하며 optimizer, scheduler, epoch와 config를 포함하지 않는다.
- training script의 `--checkpoint`는 resume load가 아니라 `--save`와 함께 사용할 save path다.
- automatic experiment name은 전체 data와 training identity를 기록하지 않는다.
- CLI seed는 data split 재현에 도움을 주지만 augmentation, initialization, GPU의 bit-level determinism을
  보장하지 않는다.
- batch runner는 common parser의 모든 option을 config에서 전달하지 않는다.
- external model은 optional dependency와 local pretrained weight가 필요할 수 있다.
- predictor는 image path와 상세 postprocess failure 원인을 저장하지 않는다.

model별 추가 제약은 각 model 문서의 failure mode와 current limitation section을 확인한다.

## 13. 문서 읽기 순서

초보자는 다음 순서로 읽는 것이 좋다.

1. [Glossary](docs/glossary.md)에서 ROI, target, raw output, loss, metric 용어를 확인한다.
2. [Model Contract](docs/architecture/01-model-contract.md)에서 common tensor 흐름을 이해한다.
3. [Model Assembly](docs/architecture/02-model-assembly.md)에서 model, network, head를 구분한다.
4. [Model Guide](docs/models/README.md)에서 표현별 model을 비교한다.
5. [Dataset Guide](docs/guides/01-dataset-format.md)와 [CLI Guide](docs/guides/02-cli-usage.md)로 실행을 준비한다.
6. [Training Workflow](docs/guides/03-training-workflow.md)로 실험을 수행한다.
7. [Loss Reference](docs/reference/01-losses.md)와 [Metric Reference](docs/reference/02-metrics.md)로 결과를
   해석한다.

전체 canonical 문서 지도와 목적별 학습 경로는 [docs/README.md](docs/README.md)에서 제공한다.

## 14. Documentation baseline

current implementation이 동작의 최우선 기준이며 root README와 `docs/` canonical 문서는 이를 설명한다.
구현과 문서가 충돌하면 source를 확인하고 같은 작업에서 문서를 갱신한다. 승인된 변경 계획과 완료 이력은
`docs/plans/`에 보존한다.

## 15. 핵심 요약

이 project는 11개 model 선택자를 common normalized corner contract로 묶는다. `--model`, `--network`,
`--head`로 assembly를 정하고, labeled CSV에서 60:20:20 split을 만들어 train, evaluate, predict script를
실행한다. 첫 작업은 image size 224와 작은 custom regression run으로 확인하고, checkpoint와 history,
metrics, predictions를 한 실험 단위로 관리한다.
