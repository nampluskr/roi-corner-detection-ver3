# Glossary

이 문서는 project 문서, CLI, source에서 반복되는 용어를 초보자 관점에서 설명한다. 같은 단어가 일반적인
machine learning 문맥과 current implementation에서 조금 다르게 쓰일 수 있으므로, 단순 번역보다 이
project에서 어떤 object와 tensor를 가리키는지에 초점을 둔다.

## 1. 용어를 읽는 방법

처음부터 모든 용어를 외울 필요는 없다. 다음 흐름을 기준으로 필요한 section을 찾아 읽는다.

```text
image and CSV
-> dataset and transform
-> model assembly
-> raw output and target
-> loss and optimization
-> postprocess and final corners
-> metric and experiment output
```

architecture의 전체 계약은 [Model Contract](architecture/01-model-contract.md), model 선택 축은
[Model Assembly](architecture/02-model-assembly.md)에서 더 자세히 설명한다.

## 2. 문제와 geometry 용어

ROI corner detection의 출발점이 되는 geometry 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| ROI | Region of Interest의 약자다. image에서 찾아내거나 후속 처리하려는 관심 영역이다. current project에서는 quadrilateral로 표현한다. |
| corner | ROI 경계를 이루는 네 꼭짓점이다. 단순 point 집합이 아니라 고정된 순서를 가진다. |
| quadrilateral | 네 변과 네 corner로 이루어진 polygon이다. current final output의 geometry 형태다. |
| polygon | 순서가 있는 vertex를 edge로 연결한 닫힌 도형이다. ROI area와 IoU를 계산할 때 사용한다. |
| vertex | polygon을 이루는 한 점이다. corner와 거의 같은 뜻이지만 GCN에서는 graph node라는 의미도 가진다. |
| edge | 인접 corner 두 개를 연결한 선분이다. ridge model에서는 각 변이 dense target channel이 된다. |
| boundary | ROI 내부와 외부를 나누는 경계다. mask contour나 ridge line에서 추정할 수 있다. |
| contour | binary mask에서 같은 object boundary를 따라 연결된 point 집합이다. segmentation postprocess에서 geometry를 복원할 때 사용할 수 있다. |
| intersection | 두 line 또는 polygon이 겹치는 부분이다. ridge와 hybrid 계열은 line intersection으로 corner를 복원할 수 있다. |
| convex | polygon 내부의 두 점을 이은 선분이 polygon 밖으로 나가지 않는 형태다. common IoU 계산은 ordered convex quadrilateral에서 가장 안정적이다. |
| degenerate polygon | duplicate corner, zero area, collinear point처럼 정상적인 면적을 정의하기 어려운 polygon이다. |
| self-intersection | polygon edge가 의도하지 않게 서로 교차하는 상태다. corner order가 틀릴 때 자주 발생한다. |

## 3. Corner 순서

current coordinate contract는 다음 순서를 사용한다.

| 약어 | 전체 이름 | tensor index | 위치 |
| --- | --- | ---: | --- |
| `TL` | top-left | 0 | 왼쪽 위 |
| `TR` | top-right | 1 | 오른쪽 위 |
| `BR` | bottom-right | 2 | 오른쪽 아래 |
| `BL` | bottom-left | 3 | 왼쪽 아래 |

이 순서는 `(B, 4, 2)` tensor의 두 번째 dimension과 CSV의 `(x1,y1)`부터 `(x4,y4)`까지의 의미를 정한다.
같은 네 위치라도 순서가 다르면 coordinate loss와 polygon metric은 다른 결과를 낸다.

## 4. Coordinate와 shape

tensor 위치와 좌표를 이해하는 데 필요한 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| pixel coordinate | image grid의 실제 위치를 pixel 단위 `(u,v)`로 나타낸 값이다. |
| normalized coordinate | image width와 height에 대한 비율 `(x,y)`다. current target과 final prediction은 일반적으로 `[0,1]`을 사용한다. |
| x coordinate | 왼쪽에서 오른쪽 방향의 위치다. normalized 값 0은 왼쪽, 1은 오른쪽 경계에 대응한다. |
| y coordinate | 위에서 아래 방향의 위치다. normalized 값 0은 위쪽, 1은 아래쪽 경계에 대응한다. |
| shape | tensor 각 dimension의 크기다. 예를 들어 `(B,4,2)`는 batch, corner, coordinate dimension을 뜻한다. |
| batch dimension | 여러 sample을 한 번에 처리하는 첫 dimension `B`다. |
| channel | 같은 spatial 위치에서 서로 다른 feature 또는 class 정보를 담는 dimension이다. RGB input은 3 channel이다. |
| spatial dimension | height와 width처럼 image grid 위치를 유지하는 dimension이다. |
| resolution | spatial grid의 크기다. input 224와 output map 56은 서로 다른 resolution이다. |
| stride | input에서 몇 pixel 이동할 때 feature map에서 한 cell 이동하는지를 나타내는 scale이다. stride 4 feature는 input의 4분의 1 resolution이다. |

normalized coordinate $(x,y)$를 width $W$, height $H$의 pixel scale로 바꾸는 개념식은 다음과 같다.

$$
u=xW, \qquad v=yH
$$

current model contract와 image size 제약은 [Model Contract](architecture/01-model-contract.md)에서 설명한다.

## 5. Data 용어

CSV에서 batch까지 이어지는 data 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| dataset | index로 sample을 읽을 수 있는 전체 data object다. current labeled dataset은 CSV와 image file을 결합한다. |
| sample | dataset의 한 항목이다. labeled case에서는 image 하나와 corner target 하나의 pair다. |
| label | 학습에서 정답으로 사용하는 값의 일반 명칭이다. current common label은 네 corner다. |
| target | loss 또는 metric이 prediction과 비교하는 정답 표현이다. common corner target과 model-specific target을 구분해야 한다. |
| CSV | comma-separated values file이다. current schema는 image path와 corner 8개 값을 row마다 기록한다. |
| split | 전체 sample을 train, valid, test 부분집합으로 나누는 과정 또는 결과다. |
| train split | parameter update에 사용하는 subset이다. current ratio는 60%다. |
| validation split | 학습 중 generalization과 early stopping을 확인하는 subset이다. current ratio는 20%다. |
| test split | 학습이 끝난 checkpoint의 최종 metric과 prediction을 만드는 subset이다. current ratio는 20%다. |
| subset | 원본 dataset index 중 일부만 가진 view다. size limit과 split 결과에 사용한다. |
| seed | random split과 sampling 결과를 반복하기 위한 시작값이다. current CLI seed가 전체 training의 bit-level 재현성을 보장하지는 않는다. |
| data stage | `public`, `synthetic`, `measured`처럼 output path에 기록하는 논리 category다. current `--dataset`은 CSV를 자동 선택하지 않는다. |

dataset 준비는 [Dataset Format Guide](guides/01-dataset-format.md)를 참고한다.

## 6. Transform과 dataloader

sample을 network input으로 바꾸는 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| transform | image와 corner를 resize, tensor conversion, normalization하는 함수 또는 함수 sequence다. |
| joint transform | image의 geometry와 corner 좌표를 함께 변경하는 transform이다. flip과 rotation이 해당한다. |
| augmentation | training data 다양성을 늘리기 위한 random 변환이다. validation과 test에는 적용하지 않는다. |
| resize | image spatial size를 변경한다. normalized corner는 비율이므로 current resize에서 값이 유지된다. |
| horizontal flip | image를 좌우 반전하고 x coordinate와 corner index를 함께 바꾼다. |
| vertical flip | image를 상하 반전하고 y coordinate와 corner index를 함께 바꾼다. |
| normalization | channel별 mean을 빼고 standard deviation으로 나누어 input scale을 맞춘다. |
| ImageNet normalization | ImageNet pretrained network에서 널리 사용하는 RGB mean과 standard deviation 기준이다. |
| dataloader | dataset sample을 batch로 묶고 shuffle, worker, memory transfer policy를 적용하는 iterator다. |
| shuffle | epoch마다 train sample 순서를 섞는 동작이다. valid와 test에는 사용하지 않는다. |
| drop last | 마지막 incomplete train batch를 버리는 정책이다. current train dataloader에서 활성화된다. |
| worker | CPU에서 sample load와 transform을 병렬 수행하는 process다. 오류 진단에는 worker 0이 유용하다. |

## 7. Model 조립 용어

CLI에서 자주 혼동되는 세 선택 축은 다음과 같다.

| 용어 | current project 정의 | 예시 |
| --- | --- | --- |
| model | corner를 어떤 target, loss, raw output, postprocess로 학습할지 정하는 package 선택자 | `reg`, `seg`, `det` |
| network | composable model의 backbone 또는 external complete architecture 이름 | `custom`, `resnet18`, `yolov8n` |
| head | model 안의 output variant 또는 detection pseudo-box 규모를 정하는 선택자 | `gap`, `mask`, `box`, `point` |

예를 들어 `--model seg --network custom --head mask`에서 `seg`는 mask 표현과 loss를, `custom`은 feature
encoder를, `mask`는 model이 허용하는 output variant를 나타낸다.

`model`과 neural network 전체를 일상적으로 부르는 model이라는 단어가 혼동될 수 있다. 이 문서 묶음에서
backtick이 있는 `model`은 가능한 한 CLI와 package 선택 축을 뜻한다.

## 8. Neural network component

composable architecture를 이루는 부품 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| backbone | image를 점차 추상적인 feature로 바꾸는 encoder network다. |
| encoder | input을 낮은 resolution, 높은 semantic feature로 압축하는 부분이다. backbone과 비슷한 뜻으로 쓰인다. |
| feature | network가 image에서 추출한 learned representation이다. channel과 spatial map 또는 token 형태일 수 있다. |
| global feature | image 전체를 하나의 vector로 요약한 feature다. `reg` gap head가 사용할 수 있다. |
| spatial feature | 위치 정보가 남아 있는 2D feature map이다. spatial regression과 GCN sampling에 사용한다. |
| stage feature | encoder의 서로 다른 resolution에서 나온 feature map이다. U-Net skip과 detection neck에 사용한다. |
| adapter | backbone 고유 output을 common `FeatureBundle` field로 변환하는 component다. |
| `FeatureBundle` | `global_feature`, `spatial_feature`, `stages`를 선택적으로 담는 common feature object다. |
| `FeatureSpec` | channel, stride, available field 같은 feature metadata와 capability를 기술한다. |
| decoder | 낮은 resolution feature를 upsample하고 shallow feature를 결합해 dense spatial feature를 복원한다. |
| skip connection | encoder의 shallow stage feature를 decoder의 같은 scale과 연결하는 경로다. |
| neck | 여러 stage feature를 detection에 적합한 multi-scale feature로 결합하는 component다. |
| head | feature를 coordinate, mask, class map 같은 task raw output으로 projection하는 마지막 component다. CLI head보다 좁은 network 부품 의미로도 사용된다. |
| capability | consumer가 요구하는 global, spatial, stages feature를 network가 제공할 수 있는지 나타내는 조건이다. |
| whole-model | library 내부 encoder, decoder, head와 native training contract를 하나로 유지하는 complete architecture다. |

component 경계는 [Source Layout](architecture/03-src-layout.md)에서 자세히 설명한다.

## 9. Model package component

각 `src/models/<model>/` package에서 사용하는 역할 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| model class | image에서 model-specific raw output을 계산하는 neural network object다. |
| preprocessor | common corner target을 mask, heatmap, pseudo-box 같은 training target으로 변환한다. |
| postprocessor | raw output을 ordered normalized final corner로 변환한다. |
| wrapper | model, preprocessor, postprocessor, optimizer, scheduler, loss, metric과 step lifecycle을 묶는다. |
| factory | CLI 문자열에 맞는 dataset, dataloader, wrapper object를 생성하는 composition helper다. |
| registry | 지원 model 문자열을 wrapper class에 대응시키는 dispatch 목록이다. current factory는 conditional branch 형태다. |
| contract | component가 주고받는 shape, 값 범위, 순서와 책임에 대한 공통 약속이다. |
| lifecycle | initialize, reset, train, validate, predict, save처럼 object가 실행되는 순서다. |

## 10. Prediction 표현

학습과 inference 사이에서 data가 어떤 상태인지 구분하는 용어는 다음과 같다.

| 용어 | 생성 주체 | 의미 |
| --- | --- | --- |
| common target | dataset | `(B,4,2)` normalized ordered corner |
| model-specific target | preprocessor | loss에 맞는 mask, heatmap, graph target 또는 detection label |
| raw output | model forward | sigmoid, threshold, decode 이전의 tensor, dictionary 또는 library object |
| final output | postprocessor | evaluator가 읽을 `(B,4,2)` corner |
| prediction | 문맥에 따라 raw 또는 final output | current evaluator와 CSV에서는 final corner를 뜻함 |
| decode | raw output의 class, offset, map을 실제 corner 후보로 바꾸는 계산 |
| fallback | postprocess가 정상 후보를 얻지 못했을 때 zero, center 같은 대체값을 반환하는 정책 |

raw output을 final corner로 오인하면 loss shape와 metric 의미를 잘못 이해하기 쉽다.

## 11. Activation 용어

network output 값을 해석할 때 필요한 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| activation | tensor에 element-wise 또는 channel-wise nonlinearity를 적용하는 함수다. |
| logit | activation 전의 범위가 제한되지 않은 score다. |
| probability | 일반적으로 0과 1 사이로 해석하는 confidence 값이다. |
| sigmoid | 각 logit을 독립적으로 0과 1 사이 값으로 바꾸는 함수다. binary mask와 multi-label class에 사용한다. |
| softmax | 여러 class logit을 합이 1인 분포로 바꾸는 함수다. library model 내부에서 사용할 수 있다. |
| threshold | probability 또는 score를 binary decision으로 바꾸는 기준값이다. |
| argmax | 가장 큰 값을 가진 index를 선택하는 연산이다. peak map에서 corner cell을 고를 때 사용한다. |
| clamp | 값이 정해진 최소와 최대를 벗어나지 않도록 제한하는 연산이다. numerical stability에 사용한다. |

## 12. Model 표현 용어

model 문서를 읽을 때 반복되는 표현은 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| regression | continuous coordinate나 offset을 직접 예측하는 방식이다. `reg`와 detection regression branch가 해당한다. |
| segmentation | 각 spatial cell을 ROI foreground 또는 background로 분류하는 방식이다. |
| binary mask | ROI 내부는 1, 외부는 0으로 표현한 2D target 또는 prediction이다. |
| rasterization | vector polygon을 discrete grid mask로 채우는 과정이다. |
| dense prediction | image grid의 모든 위치에서 score나 geometry를 예측하는 방식이다. |
| heatmap | 위치별 confidence를 담은 2D map이다. `peak`는 corner channel마다 Gaussian heatmap을 만든다. |
| Gaussian peak | 정답 point 중심이 가장 크고 거리에 따라 부드럽게 감소하는 target이다. |
| ridge | point가 아니라 line 주변에서 값이 큰 elongated Gaussian target이다. ROI edge를 표현한다. |
| detection | object 또는 point 후보의 class와 위치를 함께 예측하는 방식이다. |
| grid cell | dense detection map의 한 spatial 위치다. positive corner가 특정 cell에 assignment된다. |
| assignment | target corner나 object를 어느 prediction cell 또는 query가 담당할지 정하는 과정이다. |
| pseudo-box | point-like corner를 detector label로 만들기 위해 중심 주변에 부여한 fixed-size box다. |
| box center | detector가 예측한 bounding box의 중심이다. external detector에서 final corner 후보로 사용한다. |
| NMS | Non-Maximum Suppression의 약자다. 겹치는 detection 후보 중 score가 높은 것을 남기는 일반적인 연산이다. library model 내부 또는 postprocess에서 사용될 수 있다. |
| query | DETR에서 object 후보 하나를 담당하는 learned slot이다. matching 뒤 class와 box를 예측한다. |
| Hungarian matching | DETR prediction query와 target object 사이의 one-to-one assignment를 찾는 알고리즘이다. |
| graph | node와 edge 관계로 이루어진 구조다. GCN에서는 네 corner가 node가 된다. |
| GCN | Graph Convolutional Network의 약자다. 인접 corner 관계를 이용해 feature를 교환하고 위치를 refine한다. |
| refinement | initial prediction을 한 번에 끝내지 않고 여러 step에서 점진적으로 수정하는 과정이다. |
| hybrid | learned neural output과 classical geometry algorithm을 결합하는 방식이다. current hybrid는 mask와 geometry 복원을 연결한다. |
| classical CV | 학습된 parameter보다 threshold, contour, line fitting 같은 명시적 image geometry 연산을 중심으로 하는 computer vision 방식이다. |

model별 표현은 [Model Guide](models/README.md)에서 비교한다.

## 13. Training 용어

parameter를 학습하는 과정의 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| training | labeled batch로 loss와 gradient를 계산하고 parameter를 갱신하는 과정이다. |
| inference | target과 gradient 없이 image에서 prediction을 만드는 과정이다. |
| forward | input을 model에 통과시켜 raw output을 계산하는 단계다. |
| backward | loss gradient를 output에서 parameter 방향으로 계산하는 단계다. |
| gradient | parameter를 어느 방향으로 얼마나 바꾸면 loss가 변하는지 나타내는 미분값이다. |
| parameter | training으로 값이 갱신되는 weight와 bias tensor다. |
| epoch | train dataloader를 한 번 순회하는 단위다. drop-last 때문에 모든 split sample이 사용되지 않을 수 있다. |
| step | 일반적으로 batch 하나의 forward, backward, optimizer update 단위다. GCN refinement step과 문맥을 구분해야 한다. |
| optimizer | gradient를 사용해 parameter를 갱신하는 algorithm과 state다. current wrapper는 주로 AdamW를 사용한다. |
| learning rate | optimizer가 한 update에서 움직이는 scale이다. |
| scheduler | epoch나 validation score에 따라 learning rate를 조정한다. current wrapper는 주로 `ReduceLROnPlateau`를 사용한다. |
| warmup | 초기 epoch에 backbone을 freeze하고 task layer를 먼저 학습한 뒤 전체 network를 여는 phase다. |
| freeze | parameter의 gradient update를 끄는 것이다. |
| unfreeze | frozen parameter를 다시 trainable하게 만드는 것이다. |
| pretrained | 다른 dataset이나 task에서 먼저 학습한 weight를 초기값으로 사용하는 상태다. |
| fine-tuning | pretrained network를 current task data로 추가 학습하는 과정이다. |
| early stopping | validation monitor가 일정 epoch 동안 개선되지 않으면 학습을 중단하는 정책이다. |
| patience | early stopping이 개선 없이 기다리는 epoch 수다. |
| monitor | scheduler나 early stopping이 관찰하는 metric key다. current trainer default는 `iou`다. |

전체 lifecycle은 [Training and Inference Flow](architecture/04-training-and-inference-flow.md)를 참고한다.

## 14. Loss 용어

optimization objective를 이해하는 데 필요한 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| loss | raw output과 model-specific target 차이를 differentiable scalar로 만든 값이다. |
| loss component | class, box, BCE, Dice처럼 total loss를 구성하는 이름 있는 한 항이다. |
| loss weight | 여러 loss component를 더할 때 곱하는 상대 scale이다. |
| reduction | element별 loss를 sum, mean 또는 normalized sum으로 scalar화하는 방법이다. |
| BCE | binary cross-entropy다. binary logit과 target classification 차이를 측정한다. |
| Dice loss | probability mask와 target region overlap을 학습하는 soft loss다. |
| focal loss | easy example의 weight를 낮춰 sparse positive와 hard example에 집중하는 loss다. |
| Smooth L1 | 작은 error에는 quadratic, 큰 error에는 linear penalty를 적용하는 robust regression loss다. |
| Wing loss | 작은 coordinate error에 logarithmic penalty를 사용하는 regression loss다. |
| deep supervision | 마지막 output뿐 아니라 intermediate output에도 target loss를 적용하는 방식이다. |
| native loss | external library whole-model이 자체 architecture와 assignment 규칙으로 계산하는 loss다. |

수식과 current default는 [Loss Reference](reference/01-losses.md)를 참고한다.

## 15. Metric 용어

final corner 성능을 해석하는 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| metric | 사람이 성능을 비교하기 위해 계산하는 값이다. 일반적으로 backward에는 사용하지 않는다. |
| IoU | Intersection over Union이다. predicted와 target polygon의 교집합 면적을 합집합 면적으로 나눈다. |
| MCD | Mean Corner Distance다. sample의 네 대응 corner Euclidean distance 평균을 dataset에서 다시 평균한다. |
| MaxCD | Maximum Corner Distance다. sample마다 가장 큰 corner distance를 고른 뒤 dataset 평균을 낸다. |
| PCK | Percentage of Correct Keypoints다. 정해진 distance threshold 안에 든 corner 비율이다. |
| success rate | 모든 prediction coordinate가 finite인 sample 비율이다. geometry accuracy를 보장하지 않는다. |
| finite | NaN이나 positive, negative infinity가 아닌 numerical value다. |
| NaN | Not a Number다. geometry failure나 invalid arithmetic 결과를 나타낼 수 있다. |
| aggregation | sample별 metric을 dataset scalar로 누적하고 평균하는 과정이다. |
| running mean | 현재까지 처리한 sample의 누적 평균이다. progress와 epoch result에 사용한다. |

metric의 수식과 invalid sample 처리는 [Metric Reference](reference/02-metrics.md)를 참고한다.

## 16. Evaluation과 prediction 용어

학습 이후 작업을 구분하는 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| validation | training 중 valid split에서 loss와 wrapper metric을 계산하는 단계다. |
| evaluation | saved checkpoint를 test split에서 standalone common metric으로 평가하는 단계다. |
| prediction | checkpoint로 final corner를 만들고 sample별 target과 함께 CSV로 저장하는 단계다. |
| evaluator | final corner와 target으로 default metric bank를 계산하고 `metrics.json`을 저장하는 core object다. |
| predictor | sample별 target, prediction, finite success를 row로 만들고 `predictions.csv`를 저장하는 core object다. |
| postprocess failure | raw output은 존재하지만 corner 복원 규칙이 정상 geometry를 만들지 못한 상태다. |
| failure reason | prediction CSV에서 실패 원인을 기록하는 column이다. current implementation은 세부 원인 대신 `invalid_prediction`을 사용한다. |

## 17. Experiment와 output 용어

결과 파일과 identity를 이해하는 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| experiment | 고정된 data, assembly, training option으로 수행한 한 실행 또는 비교 단위다. |
| experiment identity | dataset, model, network, head, seed, training scale처럼 실행을 재구성하는 값의 묶음이다. |
| output directory | log, history, checkpoint, metric, prediction을 저장하는 folder다. |
| experiment name | current automatic rule의 `<model>_<network>_<head>_<dataset>` 문자열이다. |
| checkpoint | saved model state다. current `model.pth`에는 model state dictionary만 들어 있다. |
| state dictionary | PyTorch module의 named parameter와 buffer tensor mapping이다. |
| `run.log` | trainer의 timestamp epoch log다. 같은 directory 재실행 시 append될 수 있다. |
| `history.json` | train과 valid의 epoch별 named loss와 metric list다. |
| `metrics.json` | standalone test evaluator의 aggregate scalar result다. |
| `predictions.csv` | test sample별 target과 final corner prediction row다. |
| overwrite | 같은 path의 기존 file을 새 실행 결과로 교체하는 동작이다. most JSON, checkpoint, CSV에 적용된다. |
| reproducibility | 같은 조건에서 결과를 다시 만들 수 있는 정도다. current seed 하나만으로 bit-level training 재현성이 보장되지는 않는다. |

output 해석은 [Experiment Output Guide](guides/04-experiment-output.md)를 참고한다.

## 18. Source 구조 용어

code와 문서 작업에서 사용하는 구조 용어는 다음과 같다.

| 용어 | 의미 |
| --- | --- |
| package | Python module을 묶는 directory다. `src/models/seg/`가 한 예다. |
| module | import 가능한 Python file 또는 package다. |
| component | 한 가지 reusable 책임을 가진 class나 function이다. |
| core | model 종류와 독립적인 trainer, evaluator, predictor, factory 실행 계층이다. |
| utility | geometry, image, IO처럼 여러 계층이 재사용하는 독립 helper다. |
| dependency direction | 어느 package가 어느 package를 import하고 알아도 되는지 정한 방향이다. |
| composition root | 문자열 option을 실제 object graph로 조립하는 위치다. current factory와 wrapper constructor가 이 역할을 나눠 가진다. |
| canonical document | current project에서 설계와 사용 기준으로 우선하는 문서다. 구현과 충돌하면 구현을 확인하고 문서를 갱신한다. |
| plan | 실제 작업 전에 범위, 완료 기준, 검증을 기록하는 `docs/plans/` 문서다. |

## 19. 자주 혼동하는 용어 비교

비슷해 보이지만 구분해야 하는 용어는 다음과 같다.

| 용어 쌍 | 차이 |
| --- | --- |
| model과 network | model은 표현과 lifecycle package, network는 backbone 또는 whole architecture 선택이다. |
| head option과 head component | CLI head는 variant 선택값, component head는 feature를 raw output으로 바꾸는 module이다. |
| target과 final output | target은 정답, final output은 model prediction이다. 둘 다 corner shape일 수 있다. |
| raw output과 final output | raw output은 model-native 표현, final output은 postprocessed common corner다. |
| loss와 metric | loss는 학습 gradient, metric은 해석과 비교를 위한 값이다. |
| validation과 evaluation | validation은 학습 epoch 안의 valid split, evaluation은 saved checkpoint의 test split 작업이다. |
| backbone과 whole-model | backbone은 feature encoder, whole-model은 encoder부터 task output까지 완성된 architecture다. |
| decoder와 postprocessor | decoder는 learned feature module, postprocessor는 raw output을 corner로 바꾸는 inference rule이다. |
| data stage와 CSV source | data stage는 output label, CSV source는 실제 sample 목록이다. |
| success와 accuracy | success는 finite output 여부, accuracy는 IoU와 distance 같은 metric으로 판단한다. |
| warmup과 resume | warmup은 초기 freeze phase, resume는 이전 optimizer와 epoch를 이어가는 기능이다. current script는 resume를 제공하지 않는다. |

## 20. 권장 학습 순서

처음 project를 읽는 사용자는 다음 순서로 용어를 확장하는 것이 좋다.

1. ROI, corner order, normalized coordinate를 이해한다.
2. dataset, target, transform, batch를 이해한다.
3. model, network, head와 wrapper를 구분한다.
4. raw output, preprocessor, postprocessor를 연결한다.
5. loss, gradient, optimizer, epoch를 이해한다.
6. validation, checkpoint, evaluation, prediction을 구분한다.
7. IoU, MCD, MaxCD, PCK, success rate를 함께 해석한다.
8. 관심 model의 mask, dense map, detection, GCN 표현을 학습한다.

## 21. 핵심 요약

이 project의 중심 구분은 `model`, `network`, `head`, common corner와 model-specific 표현이다. dataset은
ordered normalized corner를 제공하고 preprocessor는 loss target을, model은 raw output을, postprocessor는
final corner를 만든다. wrapper와 core가 학습 및 평가 lifecycle을 실행한다. 용어가 혼동될 때는 그 값이
어느 단계에서 생성되고 어떤 shape로 다음 component에 전달되는지를 먼저 확인한다.
