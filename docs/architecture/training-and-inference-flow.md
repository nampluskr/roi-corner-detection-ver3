# Training and Inference Flow

학습, 평가, 예측은 같은 dataset과 wrapper를 사용하지만 checkpoint와 결과 파일을 다루는 방식이 다르다.
세 경로 모두 `scripts.config.parse_args`의 CLI를 사용하고, 별도 `--output_dir`이 없으면 동일한 output
directory 규칙을 따른다.

## Training Flow

학습 흐름은 다음과 같다.

```text
CSV -> train and valid split -> transform -> Dataloader -> Wrapper.train_step
-> loss and metric accumulation -> early stopping -> history.json
```

`Trainer.fit_early_stop`은 validation score를 기준으로 개선 여부를 추적한다. `--save`를 주면 training
완료 뒤 model state를 `model.pth`에 저장한다. `warmup_epochs`를 적용하는 wrapper는 초기 epoch에 backbone을
freeze한 뒤 다음 phase에서 trainable 상태와 optimizer를 갱신한다.

## Evaluation Flow

evaluation은 checkpoint가 필수다. `scripts/evaluate.py`는 model state를 불러오고 test split에서
`Wrapper.predict_step`을 호출한 뒤 `Evaluator`가 metric을 계산해 `metrics.json`으로 저장한다.

```text
checkpoint + test split -> Wrapper.predict_step -> final corners -> Evaluator -> metrics.json
```

evaluation metric은 raw output이 아니라 postprocessor 이후의 final corner를 대상으로 한다. 따라서 서로
다른 표현의 model도 동일한 normalized coordinate 기준으로 비교한다.

## Prediction Flow

`scripts/predict.py`도 checkpoint가 필수다. `Predictor`는 test batch를 순회해 target과 prediction을 같은
row에 기록하고 `predictions.csv`를 저장한다. row의 `index`는 test split 내부 순서이며 원본 image path가
아니다.

예측 CSV는 scalar metric만으로 찾기 어려운 ordering 오류, 중심 fallback, 비정상 corner를 점검하는
표본 단위 기록이다.
