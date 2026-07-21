# Metric Reference

evaluation은 postprocessor 이후의 normalized corner를 대상으로 한다. `Evaluator`는 test split 전체를
순회해 fresh metric bank를 만들고 `metrics.json`에 scalar result를 저장한다.

## Default Metric Bank

기본 evaluator는 다음 metric을 계산한다.

| key | metric | 좋은 방향 | 의미 |
| --- | --- | --- | --- |
| `iou` | polygon IoU | 높음 | predicted와 target quadrilateral의 area overlap |
| `mcd` | mean corner distance | 낮음 | 표본별 네 corner 거리의 평균 |
| `maxcd` | max corner distance | 낮음 | 표본별 가장 큰 corner 거리 |
| `pck_002` | PCK at 0.02 | 높음 | normalized distance 0.02 이내 corner 비율 |
| `pck_005` | PCK at 0.05 | 높음 | normalized distance 0.05 이내 corner 비율 |
| `sr` | success rate | 높음 | finite prediction을 만든 표본 비율 |

distance metric은 normalized coordinate에서 Euclidean distance를 계산한다. image pixel 단위 오류로
해석하려면 값을 해당 image width와 height scale에 맞춰 별도로 변환해야 한다.

## Aggregation

`BaseMetric`은 NaN이 포함된 prediction 또는 NaN metric value를 평균에서 제외한다. `SuccessRate`는
finite 여부를 모든 표본에서 직접 센다. 따라서 distance와 IoU를 볼 때는 success rate도 함께 읽어야
failure 표본이 평균에서 제외된 영향을 구분할 수 있다.

wrapper training metric은 model별로 IoU만 등록된 경우가 많고, standalone evaluator의 default bank와
동일하지 않을 수 있다. 서로 다른 experiment를 비교할 때는 학습 history 대신 같은 test split의
`metrics.json`을 기준으로 비교한다.
