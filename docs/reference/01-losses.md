# Loss Reference

이 문서는 loss가 왜 필요한지, current project의 각 loss가 어떤 tensor를 비교하는지, 수식의 각 항이 어떤
역할을 하는지 설명한다. model 문서가 한 표현의 전체 흐름을 설명한다면 이 문서는 여러 model에서 반복되는
loss 계산을 하나의 기준으로 정리한다.

## 1. Loss의 역할

neural network는 처음부터 정답을 알지 못한다. forward 결과와 target 사이의 차이를 하나의 scalar로
계산하고, 그 scalar의 gradient를 parameter까지 전달해 output이 target에 가까워지도록 갱신한다. 이
scalar가 loss다.

```text
image -> model -> raw output
corner -> preprocessor -> model-specific target
raw output + target -> loss -> backward -> parameter update
```

모든 model의 final output은 `(B, 4, 2)` corner지만 loss가 보는 표현은 다르다. `reg`는 coordinate를,
`seg`는 mask cell을, `det`는 class와 regression map을 비교한다. 따라서 서로 다른 model의 loss 숫자를
직접 비교해 성능 순위를 정할 수 없다.

## 2. Loss와 metric의 차이

loss와 metric은 모두 prediction과 target을 비교하지만 목적이 다르다.

| 구분 | loss | metric |
| --- | --- | --- |
| 주목적 | gradient를 만들어 parameter 학습 | 사람이 성능을 해석하고 model 비교 |
| 입력 | model-specific raw output과 target | postprocess 이후 final corner와 common target |
| 미분 가능성 | 일반적으로 필요 | 필요하지 않음 |
| model 간 비교 | scale이 달라 직접 비교하기 어려움 | 같은 evaluator에서는 비교 가능 |
| 사용 시점 | training과 wrapper validation | validation, standalone evaluation |

예를 들어 mask threshold와 contour 추출은 discrete 연산이라 gradient를 전달하기 어렵다. `seg`는 이
postprocess 이후 IoU를 loss로 쓰지 않고, raw mask logit에 BCE와 soft Dice를 적용한다. 최종 corner IoU는
metric으로 확인한다.

metric의 자세한 계산은 [Metric Reference](02-metrics.md)를 참고한다.

## 3. Logit과 probability

많은 head는 probability가 아니라 범위가 제한되지 않은 logit $z$를 출력한다. sigmoid는 logit을 0과 1
사이 probability $p$로 바꾼다.

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

large positive logit은 1에 가까운 probability, large negative logit은 0에 가까운 probability가 된다.

현재 loss에서 sigmoid를 적용하는 위치는 다음과 같다.

| loss | sigmoid 처리 |
| --- | --- |
| `BCELoss` | `BCEWithLogitsLoss` 내부에서 처리 |
| `DiceLoss` | loss 안에서 명시적으로 처리 |
| `BCEDiceLoss` | BCE 내부와 Dice 계산 전에 처리 |
| `FocalLoss` | probability와 stable BCE 계산에 사용 |
| `HeatmapMSELoss` | MSE 전에 처리 |
| `HeatmapFocalLoss` | focal 계산 전에 처리하고 clamp |
| `SmoothL1Loss` | detection offset의 첫 두 channel만 처리 |
| `WingLoss` | `apply_sigmoid=True`일 때 처리 |

이미 sigmoid를 적용한 값에 `BCEWithLogitsLoss`를 다시 사용하면 probability를 logit처럼 해석하므로 계산이
달라진다. model output과 loss가 어느 경계에서 sigmoid를 소유하는지 확인해야 한다.

## 4. Reduction과 running mean

loss tensor는 cell, channel, sample마다 여러 값을 가질 수 있다. reduction은 이를 backward에 사용할 scalar로
합치는 방법이다. current shared loss는 각 구현 안에서 mean이나 normalized sum을 반환한다.

`BaseLoss.__call__()`은 batch loss tensor를 반환하는 동시에 `len(target)`을 count로 사용해 다음 running
state를 갱신한다. target이 일반 tensor이면 `len(target)`은 batch sample 수 $B$다.

$$
S \leftarrow S + L_b B, \qquad C \leftarrow C + B
$$

여기서 $L_b$는 batch mean loss, $B$는 tensor target의 batch sample 수, $S$는 누적 합, $C$는 누적 sample
수다. epoch result는 다음과 같다.

$$
L_{epoch} = \frac{S}{C}
$$

tensor target을 쓰는 loss는 마지막 batch 크기가 작아도 sample 수로 가중된다. trainer는 train과 valid
epoch 시작 전에 loss state를 reset한다.

### 4.1 Dictionary target의 current aggregation

`det`의 `FocalLoss`와 `SmoothL1Loss`는 dictionary target을 받는다. Python에서 `len(dictionary)`는 batch
size가 아니라 key 수다. 따라서 이 두 shared loss의 running count는 sample 수 대신 target dictionary key
수로 증가한다. batch마다 dictionary 구조가 같으므로 reported epoch value는 사실상 batch mean을 다시
동일 weight로 평균하는 형태가 된다.

train dataloader는 incomplete batch를 버리므로 train batch 크기가 같아 영향이 작지만, validation의 마지막
batch가 작으면 정확한 sample-weighted mean과 차이가 날 수 있다. 이 차이는 history reporting에 해당하며
각 batch에서 계산한 loss tensor와 backward gradient 자체를 바꾸지는 않는다.

## 5. 여러 loss의 결합

wrapper는 이름이 있는 loss dictionary를 가진다. 각 loss의 반환값을 $L_k$, `BaseLoss.weight`를 $w_k$라고
하면 backward에 사용하는 total은 다음과 같다.

$$
L_{total} = \sum_k w_k L_k
$$

`seg`는 `bce`와 `dice` 두 object를 각각 weight 1로 더한다. `hybrid`는 `BCEDiceLoss` object 하나 안에서
BCE와 Dice를 더한 뒤 그 object의 외부 weight를 적용한다. 두 방식은 log에 표시되는 key 수가 다르다.

history에는 이름별 running loss가 기록되지만 weighted total 자체는 별도 key로 저장되지 않는다. custom
weight를 사용하면 history의 개별 숫자만 더해서 실제 total을 재구성할 때 weight를 함께 알아야 한다.

## 6. Binary cross-entropy

`BCELoss`는 binary mask의 각 cell을 foreground 또는 background classification으로 본다. target
$y \in \{0,1\}$, sigmoid probability $p$에 대한 binary cross-entropy는 다음과 같다.

$$
L_{BCE} = -\left[y\log p + (1-y)\log(1-p)\right]
$$

target이 1이면 $p$가 작을수록 큰 penalty를 받고, target이 0이면 $p$가 클수록 큰 penalty를 받는다.

current `BCELoss`는 `torch.nn.BCEWithLogitsLoss`를 사용한다. sigmoid와 logarithm을 하나의 stable 연산으로
처리하므로 raw logit을 직접 전달해야 한다. 모든 batch, channel, spatial element에 대한 기본 mean을
반환한다.

### 6.1 사용 model

기본 BCE object를 사용하는 model은 다음과 같다.

| model | raw output | target |
| --- | --- | --- |
| `seg` | `(B, 1, H_m, W_m)` mask logits | rasterized ROI mask |
| `torchseg` | `(B, 1, H_m, W_m)` whole-model mask logits | rasterized ROI mask |

두 wrapper는 BCE와 Dice를 별도 loss로 더한다. background cell 수가 foreground보다 많으면 BCE가 background
분류에 크게 영향을 받을 수 있어 Dice가 overlap 관점을 보완한다.

## 7. Soft Dice loss

Dice coefficient는 두 binary region의 overlap을 측정한다. hard threshold 대신 probability를 사용하면
gradient를 계산할 수 있는 soft Dice가 된다.

sample 하나에서 prediction probability vector를 $p$, target vector를 $y$, smoothing 값을 $s$라고 하면
current 계산은 다음과 같다.

$$
Dice(p,y) = \frac{2\sum_i p_i y_i + s}{\sum_i p_i + \sum_i y_i + s}
$$

loss는 다음과 같다.

$$
L_{Dice} = 1 - Dice(p,y)
$$

current default $s=1$이다. sample마다 spatial dimension을 flatten해 Dice를 계산한 뒤 batch mean을 반환한다.
foreground가 작아도 region overlap 전체를 직접 보므로 pixel mean BCE를 보완한다.

### 7.1 Smoothing의 의미

smoothing은 target과 prediction이 거의 비어 있을 때 denominator가 0에 가까워지는 문제를 줄인다. 그러나
$s$가 실제 foreground 규모에 비해 너무 크면 작은 mask의 차이를 약화할 수 있다. current implementation은
모든 resolution에 같은 값 1을 사용한다.

## 8. Combined BCE and Dice

`BCEDiceLoss`는 한 object 안에서 두 항을 더한다.

$$
L = L_{BCE} + \lambda_{Dice} L_{Dice}
$$

current `dice_weight` 기본값은 1이다. `hybrid` wrapper는 이 combined loss를 `loss` key 하나로 사용한다.
따라서 history에서는 BCE와 Dice component를 따로 볼 수 없다.

`seg`의 두 separate loss와 계산 목적은 비슷하지만 logging과 weight customization 경계가 다르다.

## 9. Wing loss

coordinate regression에서는 단순 L1 또는 L2 대신 작은 error에 더 민감한 penalty를 사용할 수 있다.
`WingLoss`는 absolute error $e=|\hat{y}-y|$에 대해 다음 piecewise function을 사용한다.

$$
L_{wing}(e) =
\begin{cases}
w\log\left(1+\frac{e}{\epsilon}\right), & e < w \\
e-C, & e \ge w
\end{cases}
$$

연속성을 위한 상수 $C$는 다음과 같다.

$$
C = w - w\log\left(1+\frac{w}{\epsilon}\right)
$$

current 기본값은 $w=10$, $\epsilon=2$다. `reg` wrapper는 `apply_sigmoid=True`로 raw coordinate logit을
`[0,1]` prediction으로 바꾼 뒤 normalized target과 비교한다.

### 9.1 Current normalized coordinate에서의 의미

prediction과 target이 모두 `[0,1]`이면 absolute error는 최대 1이다. 기본 $w=10$보다 작으므로 정상적인
normalized input에서는 사실상 logarithmic 첫 branch만 사용한다. linear branch는 current default scale에서
도달하지 않는다.

이 사실은 일반적인 Wing loss 설명과 current parameter scale을 구분해야 하는 이유다. pixel coordinate용
hyperparameter를 normalized coordinate에 그대로 사용한 상태로 볼 수 있으며, 적절성은 별도 실험으로
검증해야 한다.

## 10. Detection classification focal loss

`det` class map에는 대부분 background cell이고 corner가 배정된 positive cell은 매우 적다. 일반 BCE를
평균하면 많은 easy background가 gradient를 지배할 수 있다. focal loss는 이미 잘 맞힌 example의 weight를
줄여 어려운 example에 집중한다.

target class를 $y$, probability를 $p$라고 할 때 correct-class probability $p_t$는 다음과 같다.

$$
p_t = yp + (1-y)(1-p)
$$

class balancing weight는 다음과 같다.

$$
\alpha_t = y\alpha + (1-y)(1-\alpha)
$$

current focal loss는 다음 값을 모든 class와 cell에서 mean한다.

$$
L_{focal} = \alpha_t(1-p_t)^\gamma L_{BCE}
$$

기본값은 $\alpha=0.25$, $\gamma=2$다. 잘 맞힌 example은 $p_t$가 1에 가까워 $(1-p_t)^\gamma$가 작아진다.

### 10.1 Input dictionary

`FocalLoss`는 일반 tensor 두 개가 아니라 dictionary를 받는다.

```text
raw_output["cls"] -> class logits
target["cls"]     -> binary class target
```

같은 `det` raw output의 regression branch는 별도 `SmoothL1Loss`가 처리한다.

## 11. Heatmap focal loss

`peak`와 `ridge`는 Gaussian dense target을 사용한다. target 값이 정확히 1인 위치를 positive로 보고 나머지
위치는 negative로 보되, Gaussian 중심 주변의 target이 큰 negative에는 작은 weight를 준다.

probability를 $p$, target을 $y$, 기본 exponent를 $\alpha=2$, $\beta=4$라고 하면 positive와 negative 항은
다음 형태다.

$$
L_{pos} = (1-p)^\alpha \log(p)\,\mathbb{1}[y=1]
$$

$$
L_{neg} = p^\alpha \log(1-p)(1-y)^\beta\,\mathbb{1}[y\ne1]
$$

최종 loss는 두 합의 음수를 positive cell 수로 나눈다.

$$
L = -\frac{\sum L_{pos}+\sum L_{neg}}{\max(N_{pos},1)}
$$

probability는 logarithm 전에 `1e-6`부터 `1-1e-6`까지 clamp해 `log(0)`을 피한다.

### 11.1 Peak와 ridge의 current 차이

`PeakPreprocessor`는 discretized corner cell 중심에 exact target value 1을 만든다. 네 channel에 하나씩
positive가 있어 일반적으로 sample당 네 positive를 제공한다.

`RidgePreprocessor`는 line distance에서 Gaussian ridge를 만들지만 각 map을 exact maximum 1로 다시 normalize
하지 않는다. discretized grid의 어느 cell도 line 위에 정확히 놓이지 않으면 target에 exact `1.0`이 없을 수
있다. `HeatmapFocalLoss`는 `target.eq(1.0)`을 사용하므로 이 경우 $N_{pos}=0$이고 모든 cell이 negative 항으로
처리된다. denominator는 clamp로 1이 되어 division error는 없지만 positive attraction term은 사라진다.

이는 current 구현을 문서화한 것이며 자동 보정 기능이 있다는 뜻이 아니다. ridge 학습 결과를 해석할 때
target maximum과 positive count를 점검할 필요가 있다.

## 12. Heatmap MSE loss

`HeatmapMSELoss`는 sigmoid probability와 Gaussian target 사이 mean squared error를 계산한다.

$$
L_{MSE} = \frac{1}{K}\sum_i (\sigma(z_i)-y_i)^2
$$

여기서 $K$는 batch, channel, spatial element 전체 수다. 구현은 shared component로 존재하지만 current
default wrapper는 사용하지 않는다. `peak`와 `ridge`의 기본값은 `HeatmapFocalLoss`다.

MSE는 모든 cell을 균등하게 평균하므로 sparse target에서는 background가 많은 영향을 줄 수 있다. focal
loss와 비교 실험하려면 wrapper loss를 code-level에서 교체해야 하며 current CLI에는 loss selector가 없다.

## 13. Masked Smooth L1 loss

`det` regression branch는 positive cell에서만 corner offset 또는 pseudo-box geometry를 학습한다.
absolute error $e$와 transition parameter $\beta$에 대한 Smooth L1은 다음과 같다.

$$
L_{smooth}(e) =
\begin{cases}
\frac{e^2}{2\beta}, & e < \beta \\
e-\frac{\beta}{2}, & e \ge \beta
\end{cases}
$$

current 기본 $\beta=1$이다. raw box map의 첫 두 offset channel에는 sigmoid를 적용하고, target의
`pos_mask`를 곱해 positive cell 이외의 error를 0으로 만든다.

input dictionary는 다음과 같다.

```text
raw_output["box"] -> regression map
target["box"]     -> regression target map
target["pos_mask"] -> positive cell mask
```

loss sum은 positive mask count와 regression channel 수의 곱으로 나눈다. positive가 없으면 denominator를
최소 1로 clamp한다.

## 14. Deep-supervised Smooth L1

`gcn`은 initial corner와 반복 refinement 결과를 모두 출력한다. raw shape가 `(B, S, 4, 2)`이고 target
shape가 `(B, 4, 2)`이면 target을 모든 $S$ step으로 확장한다.

각 step의 Smooth L1 mean을 $L_s$, step weight를 $a_s$라고 하면 최종 loss는 다음과 같다.

$$
L = \frac{\sum_{s=1}^{S}a_s L_s}{\sum_{s=1}^{S}a_s}
$$

default `late_emphasis=False`에서는 모든 $a_s=1$이다. initial과 마지막 refinement가 같은 weight를 받는다.
`late_emphasis=True`를 code-level로 지정하면 step index가 뒤로 갈수록 큰 weight를 받는다.

모든 step에 supervision을 주면 마지막 결과만 학습할 때보다 중간 refinement도 target 방향으로 움직이도록
gradient를 받을 수 있다.

## 15. Model별 기본 loss

current wrapper의 기본 조합은 다음과 같다.

| model | history key | 기본 loss | project shared loss인가 |
| --- | --- | --- | --- |
| `reg` | `loss` | `WingLoss(apply_sigmoid=True)` | 예 |
| `seg` | `bce`, `dice` | `BCELoss`, `DiceLoss` | 예 |
| `det` | `cls`, `box` | `FocalLoss`, `SmoothL1Loss` | 예, dictionary count 주의 |
| `peak` | `focal` | `HeatmapFocalLoss` | 예 |
| `ridge` | `focal` | `HeatmapFocalLoss` | 예 |
| `gcn` | `loss` | `DeepSupervisedSmoothL1Loss` | 예 |
| `hybrid` | `loss` | `BCEDiceLoss` | 예 |
| `torchseg` | `bce`, `dice` | `BCELoss`, `DiceLoss` | 예 |
| `torchdet` | library별 key | torchvision native loss dictionary | 아니오 |
| `yolo` | `box`, `cls`, `dfl` | Ultralytics native detection loss | 아니오 |
| `detr` | native loss dictionary key | Hugging Face DETR Hungarian loss | 아니오 |

shared loss의 default `BaseLoss.weight`는 모두 1이다. external detector는 library가 total loss와 component를
계산하므로 project loss object를 backward에 사용하지 않는다. external wrapper가 running component를
기록할 때는 `len(images)`를 count로 직접 전달한다.

## 16. External whole-model loss

external detector wrapper는 native library의 training call을 common trainer lifecycle에 맞춘다.

### 16.1 Torchvision detection

training mode에서 model이 반환한 `loss_dict`의 모든 value를 더해 backward한다. 각 component scalar는
`BaseLoss` instance의 running state만 빌려 history에 기록한다. validation의 eval mode에서는 detection
prediction을 만들고 IoU를 계산하며 native validation loss를 다시 계산하지 않는다.

### 16.2 YOLO

Ultralytics model의 native loss를 사용해 backward하고 detached `box`, `cls`, `dfl` component를 running
history에 기록한다. validation에서도 native loss component와 final corner IoU를 함께 계산한다.

### 16.3 DETR

Hugging Face DETR output의 total `loss`로 backward한다. Hungarian matching과 class, box component는 library
내부에서 계산된다. wrapper는 `loss_dict`의 component를 history에 기록하고 total backward 전에 별도
project loss를 더하지 않는다. gradient norm은 current default 1.0으로 clip한다.

native loss 이름과 scale은 library architecture에 따라 달라질 수 있다. project shared loss와 직접 숫자를
비교하지 않는다.

## 17. Loss와 validation metric이 다르게 움직이는 이유

loss가 감소해도 IoU가 반드시 같은 비율로 증가하지는 않는다. 대표 원인은 다음과 같다.

| 현상 | 설명 |
| --- | --- |
| mask loss 감소, corner IoU 정체 | mask가 좋아져도 threshold와 extreme-point decode가 같은 corner를 만들 수 있음 |
| dense loss 감소, ordering 오류 | map localization과 final polygon order는 다른 단계 |
| detection cls loss 감소, corner 오차 유지 | class confidence는 좋아져도 offset regression이 부족할 수 있음 |
| GCN intermediate loss 감소, final gain 작음 | 모든 step이 평균되어 마지막 step 변화만 반영하지 않음 |
| loss finite, metric NaN failure | differentiable target는 계산됐지만 geometry postprocess가 실패할 수 있음 |

학습 진단에는 loss curve와 final corner metric, sample prediction을 함께 사용해야 한다.

## 18. Numerical stability

current implementation의 안정화 장치는 다음과 같다.

- BCE는 stable `BCEWithLogitsLoss`를 사용한다.
- Dice는 numerator와 denominator에 smoothing 1을 더한다.
- heatmap focal probability는 logarithm 전에 clamp한다.
- heatmap positive count와 detection positive count denominator는 최소 1로 clamp한다.
- Smooth L1은 작은 error에서 quadratic, 큰 error에서 linear branch를 사용한다.
- DETR은 backward gradient norm을 clip한다.

이 장치가 잘못된 target을 자동으로 수정하지는 않는다. NaN target, 범위 밖 coordinate, exact positive가 없는
ridge map 같은 semantic 문제는 별도 검사가 필요하다.

## 19. 흔한 오류

loss 관련 대표 증상과 점검 항목은 다음과 같다.

| 증상 | 가능한 원인 | 점검 순서 |
| --- | --- | --- |
| BCE가 비정상적으로 변하지 않음 | sigmoid를 model과 loss에서 중복 적용 | raw output contract 확인 |
| target와 output shape mismatch | image size 또는 decoder stride 불일치 | wrapper image size와 target shape 확인 |
| heatmap focal이 background만 학습 | exact target 1이 없음 | `target.max()`와 positive count 확인 |
| detection box loss가 0 | positive assignment가 없음 | `pos_mask.sum()` 확인 |
| total loss scale을 알 수 없음 | 여러 named loss와 weight 사용 | wrapper dictionary와 weight 확인 |
| train loss만 있고 valid loss가 0 | external wrapper의 eval semantics | wrapper `eval_step` 확인 |
| 서로 다른 model loss를 직접 비교 | 표현과 reduction scale이 다름 | common evaluator metric 사용 |
| history와 backward total이 다르게 보임 | history는 component running mean | weight와 native total 확인 |

## 20. Loss 선택과 변경

current CLI에는 `--loss` option이 없다. 기본 loss는 각 wrapper constructor에서 정한다. loss를 바꾸려면
code-level에서 wrapper에 custom loss dictionary를 전달하거나 wrapper default를 수정하고 검증해야 한다.

변경할 때는 다음 항목을 함께 확인한다.

1. raw output과 target data structure가 loss signature와 맞는가?
2. sigmoid 또는 softmax를 어느 쪽이 적용하는가?
3. spatial, channel, batch reduction이 의도와 맞는가?
4. sparse positive를 어떤 denominator로 normalize하는가?
5. `BaseLoss.weight`와 내부 component weight가 중복되지 않는가?
6. running state가 reset, update, compute lifecycle을 따르는가?
7. final corner metric이 실제로 개선되는가?

## 21. Code mapping

loss 계산을 확인할 source는 다음과 같다.

| 주제 | source |
| --- | --- |
| shared loss 구현 | `src/components/losses.py` |
| weighted total과 state lifecycle | `src/models/base/wrapper.py` |
| model별 기본 loss 조립 | `src/models/<model>/wrapper.py` |
| target 생성 | `src/models/<model>/preprocessor.py` |
| external native loss | `src/models/torchdet/wrapper.py`, `src/models/yolo/wrapper.py`, `src/models/detr/wrapper.py` |
| epoch history 저장 | `src/core/trainer.py` |

## 22. 핵심 요약

loss는 final corner가 아니라 model-specific raw output과 preprocessed target 사이의 differentiable 차이다.
BCE와 Dice는 mask, focal과 masked Smooth L1은 detection, heatmap focal은 dense map, Wing은 coordinate,
deep-supervised Smooth L1은 GCN sequence를 학습한다. shared loss는 sample-weighted running mean을 기록하고
wrapper는 weight를 적용해 total을 만든다. loss 숫자는 model 사이에서 직접 비교하지 않고, final corner
metric과 prediction을 함께 해석해야 한다.
