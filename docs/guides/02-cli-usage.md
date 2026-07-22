# CLI Usage Guide

이 문서는 training, evaluation, prediction, batch run command를 실제로 구성하는 방법을 설명한다. 세 주요
script는 같은 parser를 사용하므로 option 이름은 같지만, 어떤 option이 필수인지와 어떤 output을 만드는지는
작업마다 다르다.

## 1. 실행 환경

command는 project root와 conda environment `pytorch_env`에서 실행한다.

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260720_roi-corner-detection-ver3
```

script가 project root를 Python import path에 추가하므로 `python scripts/train.py` 형태로 실행한다. 현재
workspace의 dependency 설치 절차는 이 guide 범위에 포함하지 않는다.

## 2. 세 주요 command

작업과 진입점의 대응은 다음과 같다.

| 작업 | command | 필수 추가 option | 주요 output |
| --- | --- | --- | --- |
| 학습 | `python scripts/train.py` | labeled CSV와 compatible assembly | `history.json`, 선택적 `model.pth` |
| 평가 | `python scripts/evaluate.py` | `--checkpoint` | `metrics.json` |
| 예측 기록 | `python scripts/predict.py` | `--checkpoint` | `predictions.csv` |

evaluation과 prediction은 checkpoint 없이 실행하면 `ValueError`를 발생시킨다. training은 `--checkpoint`를
load하지 않으며, 해당 option을 주면 저장 위치로만 사용한다.

## 3. Option 분류

공통 parser의 option은 data, model assembly, training lifecycle, runtime, output으로 나눌 수 있다.

### 3.1 Data option

data 관련 option은 다음과 같다.

| option | type | 기본값 | 의미 |
| --- | --- | --- | --- |
| `--csv_path` | one or more strings | machine-specific path 목록 | labeled CSV 목록 |
| `--data_dir` | string | machine-specific path | current factory에서는 직접 사용하지 않음 |
| `--dataset` | string | `public` | output path의 논리 stage 이름 |
| `--seed` | integer | `42` | split, subset, dataloader generator seed |
| `--image_size` | integer | `224` | dataloader의 square resize edge |
| `--train_size` | integer | `2000` | split 이후 train sample limit |
| `--valid_size` | integer | `1000` | split 이후 valid sample limit |
| `--test_size` | integer | `1000` | split 이후 test sample limit |

`--dataset`은 어떤 CSV를 자동 선택하는 switch가 아니다. current implementation에서는 output directory의
첫 segment를 정하는 label이다. 실제 data source는 `--csv_path`가 정한다.

### 3.2 Model assembly option

model 조립 option은 다음과 같다.

| option | 기본값 | 의미 |
| --- | --- | --- |
| `--model` | `reg` | target, loss, decode를 포함한 model package |
| `--network`, `--net` | `custom` | backbone 또는 external whole-model 이름 |
| `--head` | `gap` | model별 output variant |

`--net`은 `--network`의 alias이며 둘 다 같은 `args.network`에 저장된다. model, network, head의 자세한
조립 원리는 [Model Assembly](../architecture/02-model-assembly.md)를 참고한다.

### 3.3 Training lifecycle option

학습 관련 option은 다음과 같다.

| option | type | 기본값 | 의미 |
| --- | --- | ---: | --- |
| `--batch_size` | integer | `4` | 한 step의 sample 수 |
| `--max_epochs` | integer | `5` | 최대 epoch 수 |
| `--patience` | integer | `3` | validation 개선 없이 기다릴 epoch 수 |
| `--warmup_epochs` | integer | `1` | wrapper에 요청할 backbone warmup 수 |

`patience`는 max epoch와 별개다. early stopping이 활성화된 경우에만 더 일찍 멈춘다. model wrapper가
`iou` metric을 제공하지 않으면 early stopping이 자동 비활성화될 수 있다.

### 3.4 Runtime과 output option

실행과 저장 관련 option은 다음과 같다.

| option | 기본값 | 의미 |
| --- | --- | --- |
| `--device` | 자동 | `cpu`, `cuda`, `cuda:0` 같은 PyTorch device 문자열 |
| `--num_workers` | `4` | dataloader worker process 수 |
| `--output_dir` | 없음 | 자동 output path 대신 사용할 directory |
| `--checkpoint` | 없음 | 평가와 예측의 load path, 학습의 선택적 save path |
| `--save` | false | 학습 종료 후 model weight 저장 여부 |

device를 생략하면 wrapper는 CUDA 가능 여부에 따라 CUDA 또는 CPU를 선택한다. data 문제를 진단할 때는
`--num_workers 0`이 유용하다.

## 4. Model과 head 조합

global parser default는 `reg/custom/gap`에만 바로 맞는다. 다른 model에는 다음과 같이 head를 명시한다.

| model | 대표 head | network 예시 |
| --- | --- | --- |
| `reg` | `gap`, `spatial` | `custom`, `resnet18`, `vit_b_16` |
| `seg` | `mask` | `custom`, `resnet18` |
| `det` | `box`, `point` | `custom`, `resnet18` |
| `peak` | `peak` | `custom`, `resnet18` |
| `ridge` | `ridge` | `custom`, `resnet18` |
| `gcn` | `gcn` | `custom`, `resnet18` |
| `hybrid` | `hybrid` | `custom`, `resnet18` |
| `torchseg` | `mask` | `fcn_resnet50` |
| `torchdet` | `box`, `point` | `fasterrcnn_resnet50_fpn` |
| `yolo` | `box`, `point` | `yolov8n` |
| `detr` | `box`, `point` | `detr_resnet50` |

표의 network는 대표 예시이며 모든 local weight가 준비되어 있다는 뜻은 아니다. 실제 지원 조합과 weight
경로는 model 문서와 source registry를 함께 확인한다.

## 5. 첫 smoke training

처음에는 custom regression으로 data와 공통 pipeline을 확인한다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --patience 1 \
  --num_workers 0 \
  --save
```

정상 종료 후 terminal에는 epoch train과 valid result, best weight restore, checkpoint 경로가 표시된다.
output directory에는 `run.log`, `history.json`, `model.pth`가 있어야 한다.

## 6. 기본 output path 계산

`--output_dir`을 생략하면 다음 경로를 사용한다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

위 smoke command에서 batch size 2, epoch 1을 사용하면 path는 다음과 같다.

```text
outputs/public/reg/custom_gap/reg_bs2_ep1_custom_gap/
```

`exp_name`에 seed, CSV, train size, patience, warmup은 들어가지 않는다. 이 값만 다르게 실행하면 같은
directory의 file을 덮어쓸 수 있으므로 중요한 비교 실험에는 `--output_dir`을 명시하는 편이 안전하다.

## 7. Model별 training 예시

project model의 대표 command는 다음과 같다. CSV와 runtime option은 필요에 맞게 추가한다.

```bash
python scripts/train.py --csv_path /data/gt.csv --model reg --network custom --head gap --save
python scripts/train.py --csv_path /data/gt.csv --model seg --network custom --head mask --save
python scripts/train.py --csv_path /data/gt.csv --model peak --network custom --head peak --save
python scripts/train.py --csv_path /data/gt.csv --model ridge --network custom --head ridge --save
python scripts/train.py --csv_path /data/gt.csv --model det --network custom --head box --save
python scripts/train.py --csv_path /data/gt.csv --model gcn --network custom --head gcn --save
python scripts/train.py --csv_path /data/gt.csv --model hybrid --network custom --head hybrid --save
```

external whole-model은 optional dependency와 local pretrained weight가 필요할 수 있다.

```bash
python scripts/train.py --csv_path /data/gt.csv --model torchseg --network fcn_resnet50 --head mask --save
python scripts/train.py --csv_path /data/gt.csv --model torchdet --network fasterrcnn_resnet50_fpn --head box --save
python scripts/train.py --csv_path /data/gt.csv --model yolo --network yolov8n --head box --save
python scripts/train.py --csv_path /data/gt.csv --model detr --network detr_resnet50 --head box --save
```

각 표현의 target과 loss 차이는 [Model Guide](../models/README.md)에서 확인한다.

## 8. Evaluation command

평가는 학습 checkpoint와 같은 assembly option을 사용한다.

```bash
python scripts/evaluate.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/custom_gap/reg_bs2_ep1_custom_gap/model.pth \
  --output_dir outputs/public/reg/custom_gap/reg_bs2_ep1_custom_gap
```

정상 종료 후 `saved metrics to .../metrics.json`과 metric dictionary가 출력된다. `--output_dir`은 checkpoint
parent와 같게 지정할 수도 있고 별도 evaluation directory로 분리할 수도 있다.

## 9. Prediction command

sample별 target과 prediction을 저장하려면 다음과 같이 실행한다.

```bash
python scripts/predict.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --dataset public \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 --test_size 4 \
  --num_workers 0 \
  --checkpoint outputs/public/reg/custom_gap/reg_bs2_ep1_custom_gap/model.pth \
  --output_dir outputs/public/reg/custom_gap/reg_bs2_ep1_custom_gap
```

정상 종료 후 `predictions.csv`가 생성된다. current CSV는 image path를 기록하지 않고 test 순서의 `index`를
기록한다.

## 10. 학습, 평가, 예측 option 일치

같은 실험을 재구성하려면 다음 값을 일치시키는 것이 중요하다.

| option | 일치가 필요한 이유 |
| --- | --- |
| `--csv_path` | 동일한 전체 sample list와 split 구성 |
| `--seed` | 동일한 split과 subset 선택 |
| `--model` | 같은 wrapper와 표현 |
| `--network` | 같은 architecture와 state key |
| `--head` | 같은 output shape와 decode |
| `--image_size` | 같은 input와 target scale |
| `--test_size` | 같은 평가 sample 수 |

batch size는 일반적으로 inference 결과의 의미를 바꾸지 않지만 default output path를 바꾼다. `max_epochs`도
evaluation 계산 자체에는 쓰이지 않지만 default experiment name에 포함된다. 결과 위치 혼동을 막으려면
학습 값을 유지하거나 `--output_dir`을 명시한다.

## 11. `--checkpoint`의 작업별 의미

동일한 option 이름이지만 script별 의미가 다르다.

| script | checkpoint가 있을 때 | 없을 때 |
| --- | --- | --- |
| `train.py` | 학습 후 해당 path에 저장, `--save`도 필요 | default output 아래 저장, `--save`도 필요 |
| `evaluate.py` | 실행 전에 weight load | 오류 |
| `predict.py` | 실행 전에 weight load | 오류 |

current training script에는 checkpoint를 load해 optimizer와 epoch를 이어가는 resume 기능이 없다.

## 12. `--image_size` 주의점

parser는 어떤 양의 integer도 받을 수 있고 dataloader는 그 크기로 image를 resize한다. 그러나 current
`get_wrapper_kwargs()`는 image size를 wrapper에 전달하지 않는다. 일부 preprocessor와 postprocessor는
내부 기본값 224를 사용한다.

따라서 non-default size는 다음 문제를 만들 수 있다.

- dense raw output과 target의 spatial shape mismatch
- normalized corner와 pseudo-box pixel scale 불일치
- postprocess coordinate scale 오류

코드 변경과 model별 검증 없이 사용하는 standard command에서는 `--image_size 224`를 유지한다.

## 13. Device와 worker 선택

device 예시는 다음과 같다.

```bash
python scripts/train.py ... --device cpu
python scripts/train.py ... --device cuda
python scripts/train.py ... --device cuda:0
```

CUDA가 없는 환경에서 `--device cuda`를 강제로 지정하면 runtime error가 발생한다. option을 생략하면 자동
선택한다.

worker 관련 문제를 진단하는 command는 다음 형태다.

```bash
python scripts/train.py ... --num_workers 0
```

문제가 해결되면 성능을 위해 worker 수를 늘릴 수 있다. worker 수가 크다고 항상 빠른 것은 아니며 storage,
CPU core, memory에 따라 달라진다.

## 14. 명시적 output directory

실험 metadata를 path에 더 자세히 남기고 싶다면 직접 경로를 지정한다.

```bash
python scripts/train.py \
  --csv_path /data/gt.csv \
  --model reg --network resnet18 --head spatial \
  --seed 42 --max_epochs 50 --save \
  --output_dir outputs/public/reg/resnet18_spatial/seed42_full
```

`--output_dir`을 지정하면 automatic naming rule 전체를 대신한다. directory가 없어도 logger와 save 함수가
생성한다.

## 15. Batch experiment

여러 config를 순차 실행하려면 `scripts/batch_config.py`의 `CONFIGS`에 dictionary를 두고 다음 command를
사용한다.

```bash
python scripts/batch_run.py --mode train
python scripts/batch_run.py --mode evaluate
python scripts/batch_run.py --mode predict
python scripts/batch_run.py --mode all
```

train mode는 각 command에 `--save`를 자동 추가한다. evaluate와 predict는 config에 checkpoint가 없으면
config로 계산한 default output 아래 `model.pth`를 찾는다.

`all` mode는 전체 config의 train을 실행한 뒤 전체 config의 evaluate, predict를 순서대로
실행한다. 개별 config가 실패해도 나머지 config와 후속 mode를 계속 실행하며, 하나라도
실패하면 최종 exit code는 1이다.

기본 `scripts/batch_config.py` 대신 `CONFIGS`를 정의한 별도 Python 파일을 사용할 때는
`--config`로 path를 지정한다.

```python
# configs/smoke_config.py

CONFIGS = [
    {
        "model": "reg",
        "network": "custom",
        "head": "gap",
        "batch_size": 2,
        "max_epochs": 1,
        "train_size": 8,
        "valid_size": 4,
        "test_size": 4,
        "num_workers": 0,
    },
]
```

```bash
python scripts/batch_run.py --mode train --config configs/smoke_config.py
python scripts/batch_run.py --mode all --config /absolute/path/to/full_config.py
```

상대 path는 current working directory에서 먼저 찾고, 없으면 project root 기준으로 찾는다. 지정한
Python 파일은 top-level `CONFIGS`를 list 또는 tuple로 정의해야 한다. `--config`를 생략하면
기존 `scripts/batch_config.py`를 사용한다.

batch runner가 전달하는 key는 `PASS_KEYS`에 제한된다. 현재 `csv_path`, `dataset`, `seed`, `image_size`는
dictionary에 넣더라도 command로 전달되지 않는다. 이 값이 필요한 batch experiment는 runner 변경이 먼저
필요하다.

## 16. 흔한 CLI 오류

대표 증상과 점검 항목은 다음과 같다.

| 증상 | 원인 후보 | 해결 방향 |
| --- | --- | --- |
| default 실행에서 CSV not found | machine-specific default path | `--csv_path` 명시 |
| head 관련 `ValueError` | global default `gap`과 model 불일치 | model별 head 명시 |
| network not supported | registry에 없는 이름 또는 capability 부족 | model 문서와 registry 확인 |
| local weight not found | pretrained file 미준비 | 요구 path와 file 확인 |
| evaluation checkpoint required | `--checkpoint` 누락 | 학습 output의 `model.pth` 지정 |
| checkpoint key mismatch | assembly 불일치 | model, network, head 일치 |
| output이 예상한 곳에 없음 | batch size나 epoch가 default path를 변경 | calculated path 또는 `--output_dir` 확인 |
| `model.pth`가 없음 | training에 `--save` 누락 | `--save`를 포함해 다시 학습 |
| non-default size shape error | wrapper에는 image size 미전달 | 224로 실행 |

## 17. Code mapping

CLI 동작은 다음 source에서 확인할 수 있다.

| 기능 | source |
| --- | --- |
| parser와 default | `scripts/config.py` |
| output naming | `scripts/config.py` |
| train command | `scripts/train.py` |
| evaluation command | `scripts/evaluate.py` |
| prediction command | `scripts/predict.py` |
| batch config | `scripts/batch_config.py` |
| batch subprocess | `scripts/batch_run.py` |
| model dispatch | `src/core/factory.py` |

## 18. 핵심 요약

세 script는 같은 option을 사용하지만 training은 parameter를 만들고, evaluation은 공통 metric을 저장하며,
prediction은 sample별 corner를 저장한다. `--model`, `--network`, `--head`를 하나의 assembly identity로
관리하고, 평가와 예측에서 학습 당시 값을 다시 지정해야 한다. 첫 실행은 명시적 CSV, image size 224,
작은 subset, worker 0으로 확인한 뒤 규모를 늘리는 것이 좋다.
