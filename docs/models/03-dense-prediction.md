# Dense Corner Prediction (`peak`, `ridge`)

`peak`와 `ridge`는 좌표를 몇 개의 숫자로 바로 출력하지 않고 2D grid 전체에 score를 예측한다. 이런
출력을 dense map이라고 한다. `peak`는 각 corner 주변에 Gaussian 봉우리를 만들고, `ridge`는 polygon의
각 변을 따라 Gaussian 능선을 만든다. 두 model은 같은 backbone, decoder, head 구조를 공유하지만 target과
postprocess가 다르다.

이 문서는 dense prediction이 필요한 이유부터 Gaussian target, focal loss, hard argmax, weighted PCA와
line intersection까지 단계적으로 설명한다.

## 1. 좌표 표현의 한계와 dense map

direct regression은 image feature를 8개 숫자로 압축한다. 이때 좌상 corner가 image의 어느 위치에서
강하게 보였는지 같은 spatial evidence는 최종 vector 안에 암묵적으로만 남는다.

dense prediction은 output grid의 각 cell에 값을 둔다. `56 x 56` map이라면 corner 하나를 표현하는 데
3,136개의 위치 score를 사용할 수 있다. network는 특정 위치가 정답과 얼마나 가까운지 spatial pattern으로
학습한다.

`peak`와 `ridge`가 답하려는 질문은 서로 다르다.

| model | 핵심 질문 |
| --- | --- |
| `peak` | 각 corner가 있을 가능성이 가장 높은 위치는 어디인가 |
| `ridge` | 각 polygon edge가 지나가는 직선은 무엇이며, 인접 직선은 어디서 만나는가 |

## 2. 공통 architecture

두 model의 neural network 구조는 동일하다.

```text
images
-> stage-returning backbone
-> CNNBackboneAdapter
-> FeatureBundle.stages
-> UNetDecoder
-> FourChannelDenseHead
-> (B, 4, Hd, Wd) logits
```

`FourChannelDenseHead`는 decoder feature에 `1 x 1` convolution을 적용해 channel을 4개로 줄인다. channel
4개의 의미는 model에 따라 다르다.

| channel | `peak` 의미 | `ridge` 의미 |
| ---: | --- | --- |
| 0 | `TL` corner | `TL`에서 `TR`로 이어지는 top edge |
| 1 | `TR` corner | `TR`에서 `BR`로 이어지는 right edge |
| 2 | `BR` corner | `BR`에서 `BL`로 이어지는 bottom edge |
| 3 | `BL` corner | `BL`에서 `TL`로 이어지는 left edge |

output resolution은 backbone의 첫 stage stride에 따라 정해진다. `image_size=224`이고 stride가 4이면
`56 x 56`, stride가 2이면 `112 x 112` map이 된다.

## 3. Gaussian의 직관

정답 위치 하나만 1이고 나머지가 모두 0인 target은 매우 희소하다. 정답 cell에서 한 칸만 벗어나도
완전히 틀린 것으로 취급하므로 학습 초기에 useful gradient를 얻기 어렵다.

Gaussian은 정답에서 가까운 위치에 높은 값을, 멀리 있는 위치에 낮은 값을 준다. 1D Gaussian의 핵심
형태는 다음과 같다.

$$
G(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right)
$$

`d`는 정답 point 또는 line까지의 거리이고, `sigma`는 봉우리나 능선의 폭이다. `sigma`가 작으면 target이
날카롭고, 크면 넓은 주변이 높은 값을 가진다.

## 4. `peak` target 생성

`PeakPreprocessor`는 corner별로 2D Gaussian map을 만든다. output width를 `Wd`, height를 `Hd`라고 하면
normalized corner `(x, y)`는 다음 map coordinate로 바뀐다.

$$
c_x = x(W_d - 1), \qquad c_y = y(H_d - 1)
$$

grid 위치 `(u, v)`의 target은 다음과 같다.

$$
T(u, v) = \exp\left(-\frac{(u-c_x)^2 + (v-c_y)^2}{2\sigma^2}\right)
$$

corner가 정수 cell 중앙에 놓이지 않으면 discrete map의 최대값이 정확히 1보다 작을 수 있다. 현재 구현은
각 channel의 최대값으로 map을 나눠 가장 높은 cell이 정확히 1이 되도록 정규화한다. 기본 `sigma`는 map
pixel 기준 2.0이다.

## 5. `peak` shape 예시

batch size가 4이고 output map이 `56 x 56`이면 target과 raw output은 모두 `(4, 4, 56, 56)`이다. 두 번째
차원의 네 channel은 corner identity를 나타낸다.

한 sample의 `TL=(0.2, 0.3)`이라면 center는 대략 다음과 같다.

```text
cx = 0.2 * 55 = 11.0
cy = 0.3 * 55 = 16.5
```

channel 0에는 `(11, 16.5)` 주변의 Gaussian peak가 생긴다. channel 1부터 3까지는 다른 corner 위치에
각각 독립적인 peak를 갖는다.

## 6. `peak` inference와 hard argmax

`PeakPostprocessor`는 raw logit에 sigmoid를 적용하고 channel마다 가장 높은 cell 하나를 찾는다. flattened
index를 `k`라고 하면 grid 좌표는 다음과 같다.

```text
grid_y = k // width
grid_x = k % width
```

그 뒤 `width - 1`, `height - 1`로 나누어 normalized coordinate로 복원한다. 이 방식은 hard argmax이므로
subpixel interpolation을 하지 않는다.

`56 x 56` map에서 한 cell 간격은 normalized coordinate 약 `1/55 = 0.0182`다. image가 224 pixel이면
약 4 pixel 간격에 해당한다. 높은 resolution map은 quantization을 줄이지만 memory와 계산량이 늘어난다.

## 7. `ridge` target 생성

`ridge`는 point가 아니라 polygon edge가 놓인 무한 직선을 target으로 만든다. channel `i`는 corner `i`와
corner `(i+1) mod 4`를 지나는 line을 나타낸다.

두 point를 `p1`, `p2`라고 하면 line direction은 `p2 - p1`이다. 이 direction에 수직인 unit normal
vector를 `n`이라고 하자. grid point `q`에서 line까지 signed distance는 다음과 같다.

$$
d(q) = (q - p_1) \cdot n
$$

ridge target은 이 거리에 Gaussian을 적용한다.

$$
T(q) = \exp\left(-\frac{d(q)^2}{2\sigma^2}\right)
$$

현재 target은 두 corner 사이의 segment에서 멈추지 않고 map 전체를 가로지르는 infinite line이다. 이
설계는 postprocessor가 line 전체의 방향을 안정적으로 fitting할 수 있게 한다.

이산 격자에서는 직선이 픽셀 중심을 정확히 지나는 일이 거의 없어 `d(q)`가 정확히 0인 픽셀이 없고,
Gaussian target의 최댓값이 1.0에 도달하지 못한다. `HeatmapFocalLoss`는 positive를 `target == 1.0`인
픽셀로 정의하므로, 정규화하지 않으면 positive anchor가 하나도 없어 loss가 배경 억제 항만 남고 ridge를
맞히는 방향으로 학습되지 않는다. 이를 막기 위해 `RidgePreprocessor`는 Gaussian을 계산한 뒤 line까지의
수직거리 `|d(q)|`가 0.5 픽셀 이하인 crest 픽셀을 정확히 1.0으로 snap한다. 그 결과 channel마다 dense한
positive line이 생겨 focal loss가 정상적으로 동작한다. `peak`가 channel별 max로 정확한 1.0 peak를
만드는 것과 같은 목적이며, ridge는 point 대신 line 전체를 positive로 사용한다는 점이 다르다.

## 8. Ridge width와 resolution

`RidgePreprocessor`는 sigma를 명시하지 않으면 `ridge_size / 28.0`으로 정한다. 그 결과 map resolution이
달라도 상대적인 ridge 폭이 유지된다.

| ridge size | default sigma | 상대 폭의 기준 |
| ---: | ---: | --- |
| 56 | 2.0 | 기준 |
| 112 | 4.0 | resolution이 두 배이므로 sigma도 두 배 |

고정 sigma를 사용하면 큰 map에서 ridge가 상대적으로 너무 얇아지고 positive 주변이 희소해질 수 있다.
현재 scaling은 pretrained backbone과 custom backbone의 stride 차이를 완화한다.

## 9. Dense map focal loss

`peak`와 `ridge`는 기본적으로 `HeatmapFocalLoss`를 사용한다. raw logit의 sigmoid probability를 `p`, target을
`y`라고 하자. target이 정확히 1인 위치는 positive이고 나머지는 negative로 처리한다.

positive에서는 `(1-p)^alpha`가 붙어 이미 잘 맞힌 위치의 영향이 줄어든다. negative에서는
`p^alpha (1-y)^beta`가 붙는다. Gaussian 중심 주변처럼 target이 1에 가까운 위치는 `(1-y)^beta`가 작아
negative penalty가 완화된다.

현재 기본값은 `alpha=2`, `beta=4`다. loss는 positive 개수로 나눈다. `PeakPreprocessor`가 각 channel의
최대값을 1로 정규화하므로 sample마다 네 positive cell이 생긴다.

`RidgePreprocessor`는 channel별 max normalization 대신 crest snap을 사용한다. continuous line이 discrete
grid cell을 정확히 통과하지 않으면 Gaussian 최댓값이 1에 도달하지 못해 `target.eq(1.0)` 조건을 만족하는
cell이 없어진다. 이 경우 positive count가 0이 되어 loss가 배경 억제 항만 남고 ridge를 맞히는 방향으로
학습되지 않는다. 이를 막기 위해 preprocessor는 line까지의 수직거리 `|d|`가 0.5 cell 이하인 crest cell을
정확히 1.0으로 설정한다. 그 결과 channel마다 line을 따라 dense한 positive cell이 생긴다. `peak`는 point
하나를 1로 정규화하고 `ridge`는 line 전체를 1로 snap한다는 점이 다르지만, 두 model 모두 sample마다 유효한
positive를 가진다는 점은 같다.

## 10. `ridge` postprocess 개요

`RidgePostprocessor`는 각 channel을 weighted point cloud로 본다. probability가 높은 map cell은 line을
fitting할 때 큰 weight를 가진다.

전체 흐름은 다음과 같다.

```text
ridge logits
-> sigmoid probabilities
-> relative background suppression
-> weighted centroid and covariance
-> principal direction for each channel
-> adjacent-line intersections
-> normalized corners
```

이 과정은 point 하나의 argmax보다 복잡하지만 edge 전체의 evidence를 사용한다.

## 11. Relative background suppression

학습이 충분하지 않은 map은 ridge뿐 아니라 background에도 낮지만 넓은 probability를 가질 수 있다.
background pixel 수가 훨씬 많기 때문에 모든 probability를 weight로 사용하면 centroid가 image 중앙으로
끌려간다.

현재 postprocessor는 channel별 peak probability를 구하고 그 값의 `rel_thresh=0.5` 미만인 pixel을 0으로
만든다. 절대 threshold가 아니라 각 channel의 최대값을 기준으로 하므로 전체 confidence가 낮은 map에서도
가장 밝은 ridge 주변은 남길 수 있다.

## 12. Weighted PCA line fitting

threshold 후 각 channel에서 weighted mean `(mean_x, mean_y)`를 계산한다. 그 다음 mean을 기준으로
`x`, `y`의 weighted covariance matrix를 만든다.

$$
\Sigma =
\begin{bmatrix}
\mathrm{cov}_{xx} & \mathrm{cov}_{xy} \\
\mathrm{cov}_{xy} & \mathrm{cov}_{yy}
\end{bmatrix}
$$

긴 ridge를 따라 point가 퍼져 있으므로 가장 큰 eigenvalue에 대응하는 eigenvector가 line의 주 direction이
된다. current implementation은 `torch.linalg.eigh`가 반환한 마지막 eigenvector를 사용한다.

각 channel은 한 point와 direction으로 표현되는 infinite line이 된다. corner `i`는 edge line `i-1`과
edge line `i`의 교점이다.

## 13. Adjacent line intersection

두 line을 `p1 + t d1`, `p2 + s d2`라고 하면 2D cross product를 이용해 `t`를 구할 수 있다. 두 direction이
거의 평행하면 denominator가 0에 가까워진다. 현재 구현은 절대값이 `1e-6` 이하인 denominator를 작은 값으로
대체한다.

이 보호는 division error를 막지만, nearly parallel line에서 매우 먼 교점이 생기는 것까지 보장하지는
않는다. ridge prediction이 잘못된 경우 normalized 범위를 벗어난 corner가 나올 수 있으므로 결과를
검증해야 한다.

## 14. `peakprod` head와 인접 채널 곱 postprocess

`ridge` model은 line-intersection 대신 peak 방식으로 corner를 복원하는 `peakprod` head를 선택할 수
있다. `--head peakprod`를 지정하면 model, target, loss는 그대로 두고 postprocessor만
`RidgePeakProductPostprocessor`로 바뀐다.

이 방식은 4개 ridge map을 하나로 합치지 않는다. ridge map은 point 봉우리가 아니라 image를 가로지르는
직선이므로, 합쳐서 단일 argmax를 하면 corner 4개를 안정적으로 얻기 어렵다. 대신 corner `i`가 line
`(i-1) mod 4`와 line `i` 위에 동시에 놓인다는 성질을 이용한다. 두 line map을 곱하면 두 직선이 만나는
교점에서만 값이 크게 남고 나머지는 억제되어 corner별 국소 peak가 만들어진다.

전체 흐름은 다음과 같다.

```text
ridge logits
-> sigmoid probabilities
-> multiply channel i by channel (i-1)%4
-> four corner peak maps
-> channel-wise hard argmax
-> normalized corners
```

`sigmoid`를 적용한 probability map을 `probs`라고 하면, 채널 정렬은 `probs.roll(1, dims=1)`로 채널 `i`
자리에 line `(i-1) mod 4`를 놓아 원소별 곱을 계산한다. 이 채널 매핑은 `RidgePostprocessor`가
`roll(1, dims=1)`로 인접 직선을 짝짓는 방식과 일치한다. 이후 corner map마다 hard argmax를 적용하고
`width - 1`, `height - 1`로 나누어 normalized corner로 복원하는 부분은 `PeakPostprocessor`와 같다.

곱 결과의 교점이 배경보다 항상 크므로 `peakprod`는 기본적으로 background threshold를 사용하지 않는다.
학습이 부족해 곱-map의 argmax가 corner를 한 점으로 몰거나 교점이 아닌 위치를 고르면 곱 전에
channel별 상대 threshold를 도입할 수 있다. `peakprod`는 nearly parallel line에서 division이 필요 없어
`RidgePostprocessor`의 교점 발산 failure mode에 덜 취약하지만, argmax quantization 한계는 `peak`와
동일하게 가진다.

## 15. 학습과 추론 흐름 비교

두 model의 흐름은 다음과 같이 비교할 수 있다.

| 단계 | `peak` | `ridge` |
| --- | --- | --- |
| target | corner-centered Gaussian | edge-line Gaussian |
| raw output | 4-channel logits | 4-channel logits |
| loss | heatmap focal | heatmap focal |
| postprocess | channel argmax | threshold, PCA, intersection |
| final output | four points | four line intersections |

같은 network와 loss를 사용해도 target과 postprocess가 다르면 학습 난이도와 failure mode가 크게 달라진다.

## 16. Warmup과 optimizer

두 wrapper는 기본 `warmup_epochs=1`을 사용한다. 첫 phase는 extractor를 freeze하고 decoder와 head를
`1e-4`로 학습한다. 두 번째 phase는 extractor `1e-5`, 나머지 component `1e-4`를 사용한다.

`ReduceLROnPlateau`는 validation IoU가 개선되지 않을 때 learning rate를 줄인다. dense loss가 감소하는
것만으로 final corner IoU가 좋아진다고 볼 수 없으므로 scheduler가 IoU를 보는 것이 중요하다.

## 17. 대표 실패 원인과 진단

`peak`의 주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| peak가 image 중앙에만 생김 | underfitting 또는 global bias | channel별 probability map |
| 두 corner channel이 같은 위치 선택 | channel identity 학습 실패 | target channel order |
| 예측이 cell 단위로만 움직임 | hard argmax quantization | map resolution |
| map이 넓고 peak가 불명확 | sigma 또는 학습 부족 | max와 second maximum 비교 |

`ridge`의 주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| 모든 교점이 중앙 근처 | background가 PCA를 지배 | threshold 전후 centroid |
| corner가 image 밖으로 크게 벗어남 | 인접 line이 거의 평행 | fitted direction과 denominator |
| top과 bottom line이 뒤섞임 | channel order 학습 실패 | ridge target channel visualization |
| custom backbone만 ridge가 약함 | resolution 대비 target 폭 또는 underfitting | sigma, max probability, epoch |

`peakprod` head는 위 실패 원인을 공유하되 교점 발산 대신 argmax 관련 증상을 보인다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| 네 corner가 한 점으로 몰림 | 곱-map이 배경에 지배됨 | 곱 전 channel별 max와 background |
| corner가 교점이 아닌 line 위를 고름 | 인접 line 학습 실패 | ridge target channel visualization |
| 예측이 cell 단위로만 움직임 | hard argmax quantization | map resolution |

## 18. Model 선택 기준

두 dense 표현의 선택 기준은 다음과 같다.

| 질문 | `peak` 선택 | `ridge` 선택 |
| --- | --- | --- |
| corner 자체가 잘 보이는가 | 적합 | 가능하지만 과도할 수 있음 |
| edge가 corner보다 연속적으로 보이는가 | local evidence만 사용 | edge 전체 evidence 사용 |
| 단순한 postprocess가 필요한가 | hard argmax | PCA와 intersection 필요 |
| subpixel refinement가 필요한가 | 현재 hard argmax 한계 | line fitting으로 보완 가능 |
| failure 분석을 단순화해야 하는가 | channel map만 보면 됨 | map과 line geometry 모두 봐야 함 |

`ridge`를 선택하되 postprocess를 단순하게 유지하고 싶으면 `peakprod` head를 사용한다. edge 지도학습은
그대로 두고 corner 복원만 인접 channel 곱과 argmax로 처리하므로, PCA와 intersection 없이 `peak`와
유사한 hard argmax 특성으로 corner를 얻는다.

## 19. Code mapping

`peak` 구현의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| dense model assembly | `src/models/peak/model.py` |
| Gaussian peak target | `src/models/peak/preprocessor.py` |
| hard argmax decode | `src/models/peak/postprocessor.py` |
| focal loss와 optimizer | `src/models/peak/wrapper.py` |

`ridge` 구현의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| dense model assembly | `src/models/ridge/model.py` |
| infinite-line ridge target | `src/models/ridge/preprocessor.py` |
| weighted PCA와 intersection | `src/models/ridge/postprocessor.py` |
| 인접 채널 곱 4-peak | `src/models/ridge/postprocessor.py` |
| head별 postprocessor 선택 | `src/models/ridge/wrapper.py` |
| focal loss와 optimizer | `src/models/ridge/wrapper.py` |

공유 decoder와 head는 `src/components/decoders.py`, `src/components/heads.py`에 있다.

## 20. 실행 예시

Gaussian peak model은 다음과 같이 실행한다.

```bash
python scripts/train.py --model peak --network custom --head peak --save
```

Gaussian ridge model은 다음과 같이 실행한다.

```bash
python scripts/train.py --model ridge --network resnet18 --head ridge --save
```

같은 ridge model을 인접 채널 곱 4-peak postprocess로 실행하려면 head를 `peakprod`로 지정한다.

```bash
python scripts/train.py --model ridge --network custom --head peakprod --save
```

`peak`의 head 이름은 고정되어 있고, `ridge`는 `ridge`와 `peakprod` 두 head를 지원한다. `peakprod`는
target과 loss를 `ridge`와 공유하고 postprocessor만 다르므로 학습된 checkpoint를 그대로 두 head로
복원해 비교할 수 있다. network를 비교할 때 output stride가 달라지면 map resolution과 target sigma도
함께 달라진다는 점을 기록한다.

## 21. 핵심 요약

`peak`는 corner 위치에 Gaussian 봉우리를 만들고 channel별 hard argmax로 점을 복원한다. `ridge`는 각
edge를 Gaussian line으로 표현하고 threshold, weighted PCA, adjacent-line intersection으로 corner를
복원한다. `ridge`는 `peakprod` head로 인접 line channel을 곱해 corner별 국소 peak를 만들고 channel별
argmax로 복원하는 대안 postprocess도 제공한다. 세 경로는 architecture와 focal loss를 공유하지만
`target이 무엇을 의미하는가`와 `postprocessor가 어떤 geometry를 사용하는가`가 다르다.
