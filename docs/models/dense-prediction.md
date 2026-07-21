# Dense Corner Prediction

`peak`와 `ridge`는 모두 4-channel dense map을 예측하지만, target이 표현하는 geometric evidence와 decode
방식이 다르다. 두 model은 같은 decoder 계열을 사용해 spatial representation과 postprocess의 차이를
비교할 수 있다.

## Peak

`peak`는 각 corner를 중심으로 한 Gaussian peak map을 target으로 사용한다.

```text
image -> extractor stages -> UNetDecoder -> FourChannelDenseHead -> peak logits
```

`PeakPreprocessor`는 corner별 Gaussian target을 만들고 `HeatmapFocalLoss`가 sparse positive를 학습한다.
`PeakPostprocessor`는 각 channel의 argmax 위치를 normalized corner로 변환한다. `--head peak`가 기본이다.

## Ridge

`ridge`는 corner의 인접 변 방향을 따라 Gaussian ridge target을 만든다. point 하나가 아니라 boundary
방향의 evidence를 학습하므로, decode는 threshold한 ridge point의 방향별 grouping과 adjacent line
intersection을 사용한다.

`RidgePreprocessor`의 default sigma는 ridge map size에 비례한다. `ridge_size = 56`에서는 기존 기준인
2.0이고, 더 큰 map에서는 상대 ridge 폭이 유지된다. 기본 loss는 `HeatmapFocalLoss`, 기본 head는 `ridge`다.

## Selection

| model | target | decode | 주된 비교 관점 |
| --- | --- | --- | --- |
| `peak` | corner-centered Gaussian | channel argmax | point localization |
| `ridge` | edge-direction Gaussian ridge | line intersection | boundary evidence |

두 model 모두 dense map resolution과 target sigma가 학습 난이도에 영향을 준다. ridge는 background
suppression threshold와 geometry decode가 추가되므로, 결과를 볼 때 raw map과 final corner를 함께 점검한다.
