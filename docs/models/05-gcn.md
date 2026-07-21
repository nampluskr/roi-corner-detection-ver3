# Graph Corner Refinement (`gcn`)

`gcn`은 네 corner를 한 번에 예측하고 끝내지 않는다. 먼저 초기 polygon을 만든 뒤, 각 corner 주변의
image feature와 이웃 corner의 정보를 사용해 위치를 여러 번 보정한다. 이때 네 corner를 graph의 vertex로
보고 Graph Convolutional Network를 적용한다.

이 문서는 graph가 왜 필요한지, 초기 corner와 spatial feature가 어떻게 결합되는지, message passing과
bounded offset이 어떻게 반복되는지 설명한다.

## 1. Corner는 서로 독립적이지 않다

quadrilateral의 네 corner는 순서와 이웃 관계를 가진다. `TL`은 `TR`, `BL`과 edge로 연결되고 `BR`과는
직접 연결되지 않는다. 한 corner가 이동하면 adjacent edge의 방향도 바뀐다.

direct regression은 8개 값을 함께 출력하므로 관계를 암묵적으로 학습할 수 있지만, 어떤 corner가
어느 corner와 정보를 교환해야 하는지 구조로 명시하지 않는다. `gcn`은 polygon cycle을 graph로 만들어
이웃 관계를 model 안에 넣는다.

핵심 질문은 다음과 같다.

> 대략적인 네 corner를 먼저 찾은 뒤, 각 위치의 local image evidence와 polygon 이웃 관계를 이용하면 더
> 정확하게 보정할 수 있는가?

## 2. Graph 기본 개념

graph는 vertex와 edge로 구성된다. 이 model에서 vertex는 corner이고 edge는 polygon의 인접 관계다.

| vertex index | corner | neighbor |
| ---: | --- | --- |
| 0 | `TL` | `TR`, `BL` |
| 1 | `TR` | `TL`, `BR` |
| 2 | `BR` | `TR`, `BL` |
| 3 | `BL` | `BR`, `TL` |

각 vertex에는 현재 coordinate와 그 위치에서 sample한 image feature가 붙는다. graph convolution은 한
vertex의 feature를 이웃 feature와 섞어 다음 hidden representation을 만든다.

## 3. 입출력 계약

`gcn`의 주요 tensor는 다음과 같다. 기본 iteration 수를 `T=3`으로 둔다.

| 단계 | shape | 의미 |
| --- | --- | --- |
| image | `(B, 3, H, W)` | RGB input batch |
| target | `(B, 4, 2)` | normalized final corner |
| spatial feature | `(B, C, Hf, Wf)` | backbone의 final feature map |
| initial corner | `(B, 4, 2)` | initial head prediction |
| vertex feature | `(B, 4, C+2)` | sampled image feature와 coordinate |
| offset | `(B, 4, 2)` | iteration별 coordinate correction |
| raw output | `(B, T+1, 4, 2)` | initial과 모든 refinement 결과 |
| final output | `(B, 4, 2)` | 마지막 step을 clamp한 corner |

기본 `T=3`이면 raw output의 두 번째 dimension은 4다. index 0은 initial corner이고 index 1부터 3은 세 번의
refinement 결과다.

## 4. 전체 architecture

현재 `GCNModel`의 흐름은 다음과 같다.

```text
images
-> CNN backbone and adapter
-> spatial feature
-> InitHead -> initial corners
-> sample local feature at each corner
-> concatenate local feature and coordinate
-> GCNRefiner -> bounded offsets
-> update corners and repeat
-> stack all corner steps
```

stage pyramid는 필요하지 않지만 spatial feature map은 필요하다. 따라서 CNN 계열 backbone을 지원하고
token-only transformer 계열은 현재 registry에 포함하지 않는다.

## 5. Initial corner prediction

`InitHead`는 spatial feature를 두 번의 strided convolution으로 줄이고 `4 x 4` adaptive pooling, flatten,
linear layer를 거쳐 8개 값을 만든다. sigmoid와 reshape를 적용해 `(B, 4, 2)` initial corner가 된다.

```text
spatial feature -> conv -> conv -> 4x4 pool -> linear(8) -> sigmoid -> four corners
```

이 단계는 `reg`의 spatial head와 비슷한 역할을 한다. initial corner가 너무 부정확하면 이후 local feature
sampling도 잘못된 위치에서 이루어지므로 refinement가 어려워진다.

## 6. Corner 위치에서 feature sampling

backbone feature map은 discrete grid지만 corner coordinate는 continuous value다. `grid_sample`은 corner
주변 네 cell을 bilinear interpolation해 해당 위치의 feature vector를 만든다.

PyTorch `grid_sample`은 coordinate 범위 `[-1, 1]`을 사용하므로 normalized corner `c`는 다음처럼 변환된다.

$$
g = 2c - 1
$$

sample 결과는 corner마다 `C` channel vector가 된다. 여기에 현재 `(x, y)` coordinate 두 값을 이어 붙여
vertex feature dimension을 `C+2`로 만든다.

coordinate가 feature map 밖으로 나가면 current implementation의 `padding_mode="border"`가 가장 가까운
border feature를 사용한다. iteration 중 corner 자체는 clamp하지 않기 때문에 이 동작이 안전장치가 된다.

## 7. Adjacency matrix

`build_normalized_adjacency`는 four-cycle edge에 self-loop를 추가한다. self-loop는 vertex가 이웃 정보뿐
아니라 자기 feature도 다음 layer에 전달하게 한다.

각 vertex는 self와 두 neighbor, 총 세 연결을 가진다. symmetric normalization은 다음 형태다.

$$
\hat{A} = D^{-1/2}(A+I)D^{-1/2}
$$

여기서 `A`는 cycle adjacency, `I`는 identity, `D`는 degree matrix다. 모든 degree가 3이므로 현재 graph에서
self와 두 neighbor의 contribution은 같은 scale로 섞인다. adjacency는 학습 parameter가 아니라 model
buffer로 저장된다.

## 8. Graph convolution과 message passing

`GCNRefiner`의 각 layer는 vertex feature에 linear projection을 적용한 뒤 adjacency를 곱하고 ReLU를
적용한다.

```text
vertex features -> Linear -> adjacency aggregation -> ReLU
```

adjacency multiplication 때문에 `TL` hidden feature에는 `TL`, `TR`, `BL`의 정보가 섞인다. layer를 여러
번 쌓으면 더 먼 vertex의 정보도 간접적으로 전달된다. 기본 `num_layers=2`, hidden dimension은 256이다.

## 9. Bounded offset

마지막 linear head는 각 vertex에 `(delta_x, delta_y)`를 출력한다. `tanh`로 값을 `[-1, 1]`에 제한한 뒤
`offset_radius`를 곱한다.

$$
\Delta c = r \tanh(o)
$$

기본 `r=0.1`이므로 한 iteration에서 각 coordinate는 최대 normalized 0.1만큼 이동한다. image size가
224라면 축 방향 최대 이동량은 약 22.4 pixel이다. bounded offset은 한 step이 polygon을 지나치게 멀리
보내는 것을 완화한다.

새 corner는 `c_next = c_current + delta_c`로 계산한다.

## 10. Iterative refinement

기본 iteration은 3회다. 각 iteration은 현재 corner 위치에서 feature를 다시 sample한다. 따라서 첫
보정으로 corner가 edge에 가까워지면 다음 iteration은 더 적절한 local evidence를 볼 수 있다.

기본 `shared_weights=True`이므로 세 iteration이 같은 `GCNRefiner` parameter를 공유한다. 동일한 correction
rule을 반복 적용하는 셈이다. constructor에서 shared weight를 끄면 iteration마다 별도 refiner를 만들 수
있지만 이 option은 공통 CLI에 노출되지 않는다.

## 11. Deep supervision

마지막 corner만 loss에 사용하면 초기 head와 중간 step은 최종 gradient를 간접적으로만 받는다. 현재
`DeepSupervisedSmoothL1Loss`는 initial을 포함한 모든 step을 같은 target과 비교한다.

기본 `late_emphasis=False`에서는 각 step loss에 같은 weight를 둔다.

$$
L = \frac{1}{T+1}\sum_{t=0}^{T} SmoothL1(c_t, y)
$$

`y`는 동일한 final corner target이다. 따라서 initial head도 직접 정답을 향해 학습하고, 각 refinement도
이전 step보다 target에 가까워지는 방향을 배운다.

## 12. Target과 preprocessor

`GCNPreprocessor`는 `(B, 4, 2)` target을 그대로 반환한다. 별도 graph label이나 iteration별 target을 만들지
않는다. loss가 하나의 target을 `(B, T+1, 4, 2)` shape로 확장해 모든 step과 비교한다.

corner 순서는 graph vertex identity이므로 매우 중요하다. target 순서가 바뀌면 adjacency가 나타내는
polygon edge도 잘못된다.

## 13. Inference와 postprocess

`GCNPostprocessor`는 raw output의 마지막 step `raw_output[:, -1]`만 선택한다. 그 뒤 coordinate를 `[0, 1]`
범위로 clamp한다.

iteration 중에는 corner가 범위를 벗어날 수 있지만 final output은 image 범위 안으로 제한된다. clamp는
값의 범위만 보장하며 corner ordering, polygon intersection, duplicate corner를 검사하지 않는다.

## 14. 단계별 예시

`TL` target이 `(0.10, 0.20)`이고 initial prediction이 `(0.16, 0.24)`라고 가정한다. 세 refinement가 다음
offset을 예측할 수 있다.

```text
initial: (0.160, 0.240)
step 1 : offset (-0.030, -0.020) -> (0.130, 0.220)
step 2 : offset (-0.018, -0.012) -> (0.112, 0.208)
step 3 : offset (-0.009, -0.006) -> (0.103, 0.202)
```

실제 offset은 네 vertex feature와 graph message를 함께 사용해 계산한다. deep supervision은 위 네 위치를
모두 target과 비교한다.

## 15. Warmup과 optimizer

기본 warmup 1 epoch 동안 extractor를 freeze하고 initial head와 GCN refiner를 `1e-4`로 학습한다. 이후
extractor는 `1e-5`, 나머지는 `1e-4`를 사용한다. scheduler는 validation IoU 정체를 감지한다.

initial prediction이 충분히 형성되지 않은 상태에서 local sampling이 의미를 갖기 어렵기 때문에 training
history에서 initial과 final step의 품질 차이를 별도로 관찰하면 진단에 도움이 된다. 현재 기본 log는
step별 metric을 직접 저장하지 않으므로 필요하면 raw output 분석이 필요하다.

## 16. 대표 실패 원인과 진단

주요 failure mode는 다음과 같다.

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| 모든 step이 같은 위치 | refiner offset이 0 근처 | iteration별 offset norm |
| refinement 후 더 나빠짐 | local feature 또는 adjacency message가 부정확 | initial과 final IoU 비교 |
| border에 corner가 붙음 | iteration 중 범위 이탈 후 final clamp | clamp 전 raw output |
| 네 점이 함께 같은 방향으로 이동 | backbone spatial bias | sampled feature와 offset 평균 |
| corner order가 꼬임 | target order 또는 큰 offset | vertex별 trajectory |
| initial이 중앙에 붕괴 | init head underfitting | `raw_output[:, 0]` 분포 |

## 17. `reg`와 비교

`gcn`은 coordinate 계열이지만 direct regression과 구조가 다르다.

| 항목 | `reg` | `gcn` |
| --- | --- | --- |
| 첫 prediction | 8 logits | 8 logits로 initial corner |
| spatial sampling | 없음 | current corner 위치에서 수행 |
| corner relation | head 안에서 암묵적 | fixed cycle adjacency로 명시 |
| output step | 한 번 | initial과 반복 refinement |
| loss | Wing | deep-supervised Smooth L1 |
| postprocess | sigmoid와 reshape | final step과 clamp |

## 18. Code mapping

개념과 source의 대응은 다음과 같다.

| 개념 | 구현 |
| --- | --- |
| initial head, adjacency, sampling, refinement | `src/models/gcn/model.py` |
| unchanged corner target | `src/models/gcn/preprocessor.py` |
| final step selection과 clamp | `src/models/gcn/postprocessor.py` |
| deep supervision과 optimizer | `src/models/gcn/wrapper.py` |
| deep-supervised loss | `src/components/losses.py` |

## 19. 실행 예시

기본 model은 다음과 같이 실행한다.

```bash
python scripts/train.py --model gcn --network custom --head gcn --save
```

pretrained CNN feature를 사용하려면 다음처럼 실행한다.

```bash
python scripts/train.py --model gcn --network resnet18 --head gcn --save
```

iteration 수, GCN layer 수, shared weight, offset radius는 constructor option이며 현재 공통 CLI에는 노출되지
않는다.

## 20. 핵심 요약

`gcn`은 spatial feature에서 initial corner를 만든 뒤, 각 corner 위치의 bilinear sampled feature와 coordinate를
graph vertex feature로 사용한다. four-cycle adjacency로 이웃 정보를 교환하고 bounded offset을 반복해서
더한다. 모든 step을 같은 target으로 학습하고 마지막 step을 final corner로 선택한다. 진단할 때는 final
결과뿐 아니라 initial과 iteration별 trajectory를 함께 보는 것이 중요하다.
