# Training Workflow

실험은 model을 바꾸기 전에 data split과 output identity를 먼저 고정해야 비교 가능한 결과가 된다.
현재 workflow는 small run으로 integration을 확인한 뒤 동일한 assembly에서 training scale을 확장하는
순서를 권장한다.

## Recommended Sequence

다음 순서로 실험을 진행한다.

1. CSV column, image path, normalized corner order를 확인한다.
2. `--train_size`, `--valid_size`, `--test_size`를 작은 값으로 두고 한 epoch training을 실행한다.
3. `--save` checkpoint를 만든 뒤 같은 assembly option으로 evaluation과 prediction을 실행한다.
4. `history.json`, `metrics.json`, `predictions.csv`를 함께 확인한다.
5. seed, data stage, model, network, head를 고정한 상태에서 epoch 또는 network를 한 축씩 변경한다.

## Model Selection

coordinate baseline은 `reg`, ROI area를 직접 학습하려면 `seg`, pseudo-box 또는 point 표현은 `det`를
사용한다. dense point representation 비교에는 `peak`와 `ridge`, 반복 정제에는 `gcn`, learned mask 뒤의
geometry 복원에는 `hybrid`를 사용한다.

`torchseg`, `torchdet`, `yolo`, `detr`은 local pretrained weight와 optional dependency가 필요할 수 있다.
실행 전에 network 이름이 지원되는지와 해당 weight가 local path에 있는지 확인한다.

## Result Interpretation

validation early stopping은 wrapper metric을 사용하고, standalone evaluation은 default metric bank를
사용한다. 따라서 experiment 비교는 같은 script와 같은 data split에서 수행한다. geometry postprocessor를
사용하는 model은 `predictions.csv`의 corner order와 중심 fallback 여부까지 확인해야 한다.
