# Experiment Output

기본 experiment output directory는 configuration에서 model assembly와 training scale을 반영해 만든다.
별도 `--output_dir`을 지정하면 이 규칙 대신 해당 경로를 사용한다.

## Directory Rule

기본 경로는 다음과 같다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

`network_head`는 `<network>_<head>`이고, `exp_name`은
`<model>_bs<batch_size>_ep<max_epochs>_<network>_<head>`다. 예를 들어 public data에서 custom `reg`의
`gap` head를 batch size 4, 5 epoch으로 실행하면 model assembly와 training scale이 경로에 함께 남는다.

## Files

실행 단계에 따라 output directory에는 다음 파일이 생성될 수 있다.

| 파일 | 생성 단계 | 내용 |
| --- | --- | --- |
| `run.log` | trainer, evaluator, predictor | timestamp가 있는 실행 log |
| `history.json` | training | epoch별 train과 valid 결과 |
| `model.pth` | training with `--save` | model state dictionary |
| `metrics.json` | evaluation | test split scalar metric |
| `predictions.csv` | prediction | 표본별 target과 final corner prediction |

같은 output directory에 서로 다른 checkpoint를 덮어쓰지 않도록 experiment name 또는 `--output_dir`을
명확히 구분한다. `metrics.json`의 숫자만 보관하지 말고 checkpoint와 prediction CSV를 함께 보존해야
결과를 재현하고 오류 표본을 확인할 수 있다.
