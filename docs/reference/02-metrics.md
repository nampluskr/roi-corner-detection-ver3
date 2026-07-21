# Metric Reference

이 문서는 final corner prediction을 어떤 기준으로 평가하는지, 각 metric의 수식과 숫자가 무엇을 뜻하는지,
invalid prediction이 평균에 어떤 영향을 주는지 설명한다. current project의 standalone evaluator는 모든
model을 동일한 normalized corner contract로 바꾼 뒤 여섯 metric을 계산한다.

## 1. Metric이 필요한 이유

model마다 training loss가 보는 표현이 다르다. `seg`의 mask BCE와 `reg`의 Wing loss를 직접 비교해 어느
model의 corner가 더 정확한지 알 수 없다. postprocessor 이후에는 모든 model이 `(B, 4, 2)` corner를
반환하므로 common metric을 사용할 수 있다.

```text
model-specific raw output
-> model-specific postprocessor
-> common normalized corners
-> common evaluator metrics
```

metric은 gradient를 만들기 위한 값이 아니라 결과를 설명하고 비교하기 위한 값이다. loss와의 차이는
[Loss Reference](01-losses.md)를 참고한다.

## 2. 평가 input contract

한 sample의 prediction과 target은 각각 `(4, 2)` array다. 순서는 `TL`, `TR`, `BR`, `BL`이고 coordinate는
`[0,1]` normalized 값이다.

| symbol | 의미 |
| --- | --- |
| $p_i=(x_i,y_i)$ | index $i$의 predicted corner |
| $t_i=(x_i^*,y_i^*)$ | 같은 index의 target corner |
| $P$ | predicted quadrilateral polygon |
| $T$ | target quadrilateral polygon |
| $N$ | 평가 sample 수 |

metric은 point ordering을 자동으로 수정하지 않는다. 같은 물리적 네 점이라도 index가 다르면 corner
distance와 polygon geometry가 달라진다.

## 3. Standalone default metric bank

`Evaluator`가 생성하는 기본 metric은 다음과 같다.

| JSON key | class | 좋은 방향 | 핵심 질문 |
| --- | --- | --- | --- |
| `iou` | `PolygonIoU` | 높을수록 좋음 | ROI polygon이 얼마나 겹치는가 |
| `mcd` | `MeanCornerDistance` | 낮을수록 좋음 | 네 corner가 평균적으로 얼마나 떨어졌는가 |
| `maxcd` | `MaxCornerDistance` | 낮을수록 좋음 | 한 sample의 가장 나쁜 corner는 얼마나 떨어졌는가 |
| `pck_002` | `PCK(0.02)` | 높을수록 좋음 | 엄격한 threshold 안에 든 corner 비율은 얼마인가 |
| `pck_005` | `PCK(0.05)` | 높을수록 좋음 | 완화된 threshold 안에 든 corner 비율은 얼마인가 |
| `sr` | `SuccessRate` | 높을수록 좋음 | finite prediction을 만든 sample 비율은 얼마인가 |

evaluation마다 `build_default_metrics()`가 fresh instance를 만든다. 이전 evaluation의 state가 다음 결과에
섞이지 않는다.

## 4. Stateful aggregation

`BaseMetric`은 `reset`, `update`, `compute` lifecycle을 사용한다.

```text
reset at evaluation start
-> update once per sample
-> accumulate total and count
-> compute total / count
```

sample $n$의 metric value를 $m_n$이라고 하면 일반적인 dataset result는 다음과 같다.

$$
M = \frac{1}{C}\sum_{n \in V}m_n
$$

$V$는 current metric이 valid하다고 받아들인 sample 집합이고 $C=|V|$다. NaN 처리 때문에 metric마다 $V$가
달라질 수 있다.

## 5. Corner Euclidean distance

대응하는 corner 한 쌍의 normalized Euclidean distance는 다음과 같다.

$$
d_i = \lVert p_i-t_i\rVert_2
= \sqrt{(x_i-x_i^*)^2+(y_i-y_i^*)^2}
$$

x와 y가 모두 normalized이므로 distance도 unit square 좌표 단위다. $d_i=0$이면 해당 corner가 정확히
일치한다.

### 5.1 Pixel scale로 해석하기

square 224 input에서 x 또는 y 한 축으로만 0.02 차이가 나면 약 4.48 pixel, 0.05 차이면 약 11.2 pixel이다.
두 축이 함께 다르면 Euclidean 관계를 사용해야 한다.

원본 image가 non-square이면 normalized x와 y를 같은 weight로 더한 distance가 원본 pixel distance와
동일하지 않다. 원본 pixel 오차가 필요하면 각 축에 width와 height를 다시 곱한 뒤 계산해야 한다.

## 6. Mean Corner Distance

sample 하나의 mean corner distance는 다음과 같다.

$$
MCD_n = \frac{1}{4}\sum_{i=1}^{4}d_{n,i}
$$

dataset `mcd`는 sample별 $MCD_n$의 평균이다.

MCD는 네 점의 평균적인 localization 품질을 한 숫자로 요약한다. corner 하나가 크게 틀리고 나머지 세
개가 정확하면 평균에 희석될 수 있으므로 MaxCD와 함께 본다.

### 6.1 간단한 예

한 sample의 corner distance가 `[0.01, 0.02, 0.03, 0.10]`이면 MCD는 다음과 같다.

$$
MCD = \frac{0.01+0.02+0.03+0.10}{4}=0.04
$$

평균 0.04만 보면 마지막 corner의 큰 오류 0.10이 잘 드러나지 않는다.

## 7. Maximum Corner Distance

sample 하나의 maximum corner distance는 다음과 같다.

$$
MaxCD_n = \max_i d_{n,i}
$$

dataset `maxcd`는 sample별 maximum을 다시 평균한다. 앞의 예에서는 `MaxCD=0.10`이다.

MaxCD는 네 corner 중 하나라도 크게 빗나가는 sample에 민감하다. document rectification처럼 한 점의 큰
오류가 전체 perspective transform을 망칠 수 있는 작업에서 중요하다.

이 값은 dataset 전체에서 가장 큰 하나의 error가 아니다. 각 sample의 최대값을 구한 뒤 그 최대값들을
평균한 결과다.

## 8. PCK

PCK는 Percentage of Correct Keypoints의 약자다. corner distance가 threshold $\tau$ 이하이면 correct로
센다.

sample 하나의 PCK는 다음과 같다.

$$
PCK_n(\tau) = \frac{1}{4}\sum_{i=1}^{4}\mathbb{1}[d_{n,i}\le\tau]
$$

dataset result는 sample별 PCK 평균이다. 모든 sample이 corner 네 개를 가지므로 전체 correct corner 수를
$4N$으로 나눈 값과 같다.

current evaluator는 두 threshold를 사용한다.

| key | threshold | square 224에서 한 축 차이의 대략적 scale |
| --- | ---: | ---: |
| `pck_002` | 0.02 | 4.48 pixel |
| `pck_005` | 0.05 | 11.2 pixel |

두 PCK 사이 차이가 크면 많은 corner가 대략 맞지만 엄격한 0.02 기준에는 들지 못한다는 뜻일 수 있다.

## 9. Polygon IoU

corner는 ROI polygon을 정의한다. predicted polygon $P$와 target polygon $T$의 Intersection over Union은
다음과 같다.

$$
IoU(P,T) = \frac{|P \cap T|}{|P \cup T|}
= \frac{|P \cap T|}{|P|+|T|-|P \cap T|}
$$

완전히 같으면 1, 겹치지 않으면 0이다. union area가 0 이하이면 current implementation은 0을 반환한다.

### 9.1 Current geometry 계산

`PolygonIoU`는 Sutherland-Hodgman polygon clipping으로 predicted polygon을 target edge마다 clip해 intersection
polygon을 만든다. area는 shoelace formula의 absolute value로 계산한다.

ordered convex quadrilateral과 일관된 방향을 전제로 할 때 가장 안정적이다. corner 순서가 잘못되거나
self-intersection, duplicate point, degenerate polygon이 있으면 일반적인 ROI overlap과 다른 값이 나올 수
있다. metric은 geometry를 자동 repair하거나 coordinate를 `[0,1]`로 clamp하지 않는다.

### 9.2 IoU와 corner distance의 차이

같은 point distance라도 ROI 크기와 shape에 따라 IoU 감소량이 다르다. 큰 polygon은 작은 위치 오차에 덜
민감할 수 있고, 작은 polygon은 같은 오차로 overlap이 크게 줄 수 있다. 반대로 polygon overlap이 높아도
특정 corner 하나가 틀릴 수 있다.

IoU, MCD, MaxCD를 함께 보는 이유가 여기에 있다.

## 10. Success Rate

`SuccessRate`는 한 prediction의 모든 coordinate가 finite인지 확인한다.

$$
success_n = \mathbb{1}[\text{all coordinates of }p_n\text{ are finite}]
$$

dataset success rate는 다음과 같다.

$$
SR = \frac{1}{N}\sum_{n=1}^{N}success_n
$$

NaN 또는 positive, negative infinity가 하나라도 있으면 실패다. `SuccessRate.update()`는 다른 metric과
달리 모든 sample을 count한다.

### 10.1 Success가 accuracy는 아니다

다음 prediction도 모두 finite이므로 success다.

- 네 corner가 모두 `(0,0)`인 zero fallback
- 네 corner가 모두 `(0.5,0.5)`인 center fallback
- `[0,1]` 범위를 벗어난 finite coordinate
- 순서가 뒤바뀐 finite corner
- 면적이 0인 duplicate corner

따라서 `sr=1.0`은 numerical output을 만들었다는 뜻이지 정확하거나 valid한 polygon이라는 뜻이 아니다.

## 11. NaN과 infinity aggregation

current `BaseMetric.update()`는 prediction에 NaN이 있으면 sample을 건너뛴다. 계산된 metric value가 Python
float NaN이어도 건너뛴다. 이 sample은 해당 metric의 numerator와 denominator 모두에 포함되지 않는다.

그러나 base check는 `np.isnan`을 사용하며 `np.isfinite`를 사용하지 않는다. infinity만 있는 prediction은
초기 skip 조건을 통과한다. 이후 동작은 metric 계산 결과에 따라 다를 수 있다.

- distance는 infinity가 되어 running total에 포함될 수 있다.
- polygon 계산이 NaN을 만들면 그 metric에서는 skip될 수 있다.
- success rate는 `np.isfinite`를 사용하므로 정확히 failure로 센다.

따라서 invalid output이 있는 experiment에서 일반 metric의 effective sample count가 서로 다를 수 있다.
current `metrics.json`은 metric별 count를 저장하지 않으므로 `sr`과 `predictions.csv`를 함께 확인해야 한다.

## 12. NaN 제외가 만드는 선택 편향

10개 sample 중 어려운 2개가 NaN이고 나머지 8개만 정확하다고 가정하자. IoU는 성공한 8개에서만 평균되어
높게 보일 수 있지만 success rate는 0.8이다.

```text
iou: valid subset 8 samples only
sr: all 10 samples
```

실패가 많은 model의 IoU를 실패가 없는 model의 IoU와 숫자 하나로만 비교하면 실패 sample이 제외된 model을
과대평가할 수 있다. 최소한 `(iou, sr)` pair로 보고 sample row를 확인한다.

## 13. Wrapper validation metric과 standalone metric

대부분의 wrapper는 training validation에 `iou`만 등록한다. `hybrid`는 `iou`와 `sr`를 등록한다. standalone
`Evaluator`는 wrapper metric dictionary를 재사용하지 않고 여섯 metric의 fresh bank를 사용한다.

| 위치 | 일반 metric | 사용 목적 |
| --- | --- | --- |
| trainer validation | wrapper별 `iou`, 일부 `sr` | scheduler와 early stopping, epoch 관찰 |
| standalone evaluator | `iou`, `mcd`, `maxcd`, 두 PCK, `sr` | test 비교와 최종 기록 |

trainer의 default early stopping monitor는 `iou`다. standalone `metrics.json`을 학습 중 monitor로 다시
사용하는 것은 아니다.

## 14. Metric reset과 state 누적

trainer는 train과 valid 시작 전에 wrapper metric을 reset한다. 따라서 같은 epoch 안에서도 train result와
valid result는 별도 state다. 다음 epoch 시작 시에도 이전 epoch 값이 제거된다.

standalone evaluator도 test 시작 전에 각 metric을 reset한다. custom code에서 같은 metric instance를 직접
재사용한다면 새 dataset 전에 `reset()`을 호출해야 한다.

## 15. Metric 예시 해석

다음 결과를 가정한다.

```json
{
  "iou": 0.78,
  "mcd": 0.025,
  "maxcd": 0.061,
  "pck_002": 0.58,
  "pck_005": 0.87,
  "sr": 0.96
}
```

이 결과는 다음과 같이 읽을 수 있다.

1. finite output을 만들지 못한 sample이 약 4% 있다.
2. valid subset의 평균 polygon overlap은 0.78이다.
3. average corner error는 0.025지만 sample의 worst corner 평균은 0.061로 더 크다.
4. corner 58%만 strict 0.02 안에 들지만 87%는 0.05 안에 든다.
5. 일부 corner가 중간 오차 구간에 있고, worst corner와 failure sample을 별도로 확인해야 한다.

이 숫자만으로 어떤 image에서 실패했는지는 알 수 없다. `predictions.csv`와 시각화가 필요하다.

## 16. Model 비교 순서

같은 test split에서 두 model을 비교할 때 다음 순서를 권장한다.

1. `sr`로 numerical failure 비율을 비교한다.
2. `iou`로 ROI overlap을 비교한다.
3. `mcd`와 `maxcd`의 차이로 corner error 분포를 본다.
4. `pck_002`와 `pck_005`로 threshold별 품질을 본다.
5. prediction CSV에서 center, zero fallback과 ordering 오류를 확인한다.
6. 같은 CSV, seed, test size, image size를 사용했는지 확인한다.

model-specific training loss는 이 비교 표에 넣지 않는다.

## 17. 공정한 비교 조건

metric 숫자가 비교 가능하려면 다음 조건을 고정한다.

| 조건 | 달라질 때 생기는 문제 |
| --- | --- |
| CSV 목록과 row order | test sample pool과 split이 달라짐 |
| seed | split과 subset selection이 달라짐 |
| test size | 평가 sample 수와 난이도가 달라질 수 있음 |
| corner order | 대응 point와 polygon 방향이 달라짐 |
| normalized coordinate convention | distance scale이 달라짐 |
| postprocessor setting | 같은 raw output도 final corner가 달라짐 |

batch size는 일반적으로 final metric 정의를 바꾸지 않지만, numerical implementation이나 default output path는
달라질 수 있다.

## 18. Metric이 말하지 않는 것

current metric bank는 다음 항목을 직접 측정하지 않는다.

- inference latency와 throughput
- model parameter 수와 memory 사용량
- confidence calibration
- 원본 image pixel 단위 error
- polygon convexity와 self-intersection 비율
- fallback 유형별 failure count
- document rectification 이후 품질
- dataset source별 subgroup 성능

필요하면 별도 metric을 추가해야 하며, 현재 지원되는 결과처럼 문서에서 가정해서는 안 된다.

## 19. 흔한 해석 오류

대표적인 오류와 올바른 확인 방향은 다음과 같다.

| 잘못된 해석 | 올바른 확인 |
| --- | --- |
| IoU가 높으면 모든 corner가 정확함 | MaxCD와 sample prediction 확인 |
| MCD가 낮으면 failure가 없음 | SR과 NaN 제외 확인 |
| SR이 1이면 valid polygon임 | zero, center, out-of-range, ordering 확인 |
| PCK 0.05가 높으면 strict accuracy도 높음 | PCK 0.02와 차이 확인 |
| MaxCD는 dataset의 단일 worst error임 | sample별 max의 dataset 평균임 |
| 같은 checkpoint면 metric 비교가 공정함 | test split과 postprocessor option도 확인 |
| training history IoU와 test IoU가 같아야 함 | split과 metric lifecycle이 다름 |

## 20. Failure 진단

metric pattern별로 먼저 확인할 항목은 다음과 같다.

| pattern | 가능한 해석 | 다음 확인 |
| --- | --- | --- |
| high IoU, high MaxCD | 한 corner가 크게 틀리지만 polygon area는 유지 | prediction corner별 distance |
| low IoU, moderate MCD | 작은 ROI 또는 polygon ordering 문제 | image별 polygon visualization |
| high PCK 0.05, low PCK 0.02 | 대략 localization하지만 정밀도가 부족 | dense resolution과 postprocess |
| low SR, good IoU | failure sample이 IoU 평균에서 제외 | failed prediction rows |
| SR 1, very low IoU | finite fallback 또는 systematic bias | repeated zero, center coordinate |
| MCD보다 MaxCD가 매우 큼 | 특정 corner 또는 class가 취약 | corner index별 error analysis |

## 21. 새 metric을 추가할 때

shared metric은 sample 하나의 prediction과 target에서 float를 계산하고 state lifecycle을 따라야 한다. 새
metric을 추가할 때 다음 항목을 확인한다.

1. input shape와 corner order를 명시한다.
2. normalized 또는 pixel 단위를 명시한다.
3. 좋은 방향이 max인지 min인지 정한다.
4. NaN, infinity, degenerate polygon 정책을 정한다.
5. sample 평균과 global aggregation 중 무엇인지 정한다.
6. result와 함께 effective count가 필요한지 검토한다.
7. `build_default_metrics()`에 넣을지 wrapper validation에만 넣을지 구분한다.

current trainer early stopping에서 사용하려면 wrapper metric key와 monitor key도 일치해야 한다.

## 22. Code mapping

metric 계산과 사용 위치는 다음과 같다.

| 주제 | source |
| --- | --- |
| stateful metric과 distance | `src/components/metrics.py` |
| polygon area | `src/utils/geometry.py` |
| default evaluator bank | `src/core/evaluator.py` |
| wrapper metric update | `src/models/base/wrapper.py` |
| validation과 early stopping | `src/core/trainer.py` |
| sample success와 CSV | `src/core/predictor.py` |
| aggregate JSON 저장 | `src/core/evaluator.py` |

## 23. 핵심 요약

standalone evaluator는 final normalized corner에 IoU, MCD, MaxCD, PCK 0.02, PCK 0.05, success rate를
적용한다. IoU는 area, MCD는 평균 point error, MaxCD는 sample의 worst corner, PCK는 threshold accuracy,
SR은 finite output 비율을 본다. NaN sample은 일반 metric 평균에서 제외될 수 있고 infinity 처리도 metric마다
달라질 수 있으므로 SR과 prediction CSV를 함께 읽어야 한다. 공정한 비교에는 같은 data split과 corner
contract가 필요하다.
