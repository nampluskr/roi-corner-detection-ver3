---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: ver3 `src/models/gcn/model.py`, `src/models/gcn/preprocessor.py`, `src/models/gcn/postprocessor.py`, `src/models/gcn/wrapper.py`, `src/models/gcn/__init__.py`, `src/components/losses.py`(신규 손실 클래스 추가), `src/core/factory.py`(dispatch 추가), `scripts/config.py`·`scripts/batch_config.py`(model 목록 추가)
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0009-peak-ridge-naming-plan.md](0009-peak-ridge-naming-plan.md), [0010-method-to-model-and-network-arg-plan.md](0010-method-to-model-and-network-arg-plan.md), [Polygon GCN 방법론 문서](https://github.com/nampluskr/roi-corner-detection/blob/main/docs/models/07_polygon-gcn.md)
---

## 목적과 배경

기존 method들은 코너 4개를 독립된 좌표(직접 회귀: `reg`) 또는 독립된 공간 표현(점 중심
가우시안: `heatmap`/`peak`, 변 중심 가우시안: `linemap`/`ridge`)으로 예측하며, 네 코너
사이의 구조적 관계(볼록성, 변의 직선성)는 학습 데이터를 통해서만 암묵적으로 반영된다.

`gcn` method는 이 관계를 명시적인 그래프로 표현한다. 코너 4개를 정점(TL, TR, BR, BL),
사각형의 네 변을 간선으로 하는 순환 그래프로 보고, 그래프 합성곱 신경망(GCN)이 정점 간
정보를 교환하며 좌표를 반복적으로 정제(coarse-to-fine)한다. 절차는 다음과 같다.

1. CNN backbone이 이미지에서 특징 맵 $F$를 추출한다.
2. 간단한 회귀 head가 $F$에서 초기 코너 추정 $Y^{(0)}$을 만든다(직접 회귀와 동일한 공간
   보존 방식).
3. $T$회(1-3) 반복: 각 정점의 현재 좌표에서 $F$를 쌍선형 보간으로 샘플링하고, 좌표를
   이어붙여 정점 특징을 만든 뒤, 고정 인접행렬(4-순환) 기반 GCN 계층을 통과시켜 정점별
   이동량(offset)을 예측하고 좌표를 갱신한다.
4. 모든 반복 단계($t = 0, \dots, T$)에 SmoothL1 손실을 걸어 심층 감독(deep supervision)한다.
5. 후처리는 마지막 반복의 좌표를 `[0, 1]`로 클램프하는 것뿐이며, 정점 순서가 그래프 위치로
   고정되어 정렬이나 미검출 처리가 필요 없다.

상세 이론(수식, 손실 함수, 3단계 학습 전략, F1-F8 제약 대응)은 위 "관련 문서"의 Polygon GCN
방법론 문서를 따른다. 이 플랜은 그 이론을 ver3 코드 구조(`BaseModel`/`BaseWrapper` 인터페이스,
`src/components/*` 재사용 컴포넌트)에 매핑하는 설계를 확정한다.

## 이론 요약과 코드 매핑

| 이론 요소 | 코드 매핑 |
|---|---|
| CNN backbone $F = \mathrm{Backbone}(I)$ | `ridge`/`heatmap`과 동일한 `custom`/`resnet`/`efficientnet`/`swin`/`vgg`/`timm-cnn` 선택 로직, `FeatureExtractor` + `CNNBackboneAdapter`(단일 최종 특징 맵 기준, `keep_spatial=True`, `keep_stages=False`) 재사용 |
| 초기 추정 head (stride conv x2 + Flatten + FC + sigmoid) | `gcn` 전용 신규 구현. `reg` method의 공간 보존 head와 동일한 사상(GAP 대신 stride 합성곱으로 격자 유지)이나, 기존 head가 이 정확한 조합인지 확인되지 않아 이 플랜에서는 `src/methods/gcn/model.py` 내부에 로컬로 구현하고 공유 컴포넌트화는 보류한다 |
| 정점 특징 샘플링(쌍선형 보간) | `torch.nn.functional.grid_sample` 기반 헬퍼를 `src/methods/gcn/model.py`에 로컬 구현(4정점·순환 그래프에 특화되어 재사용 범위가 좁으므로 컴포넌트로 분리하지 않음) |
| GCN 정제 모듈 (고정 4-순환 인접행렬, $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ 정규화, $L$계층) | `src/methods/gcn/model.py`에 `GCNRefiner`(가칭) 클래스로 로컬 구현. 인접행렬은 버퍼로 고정 등록(학습 파라미터 아님) |
| offset head ($\rho \tanh$) | 동일 파일 내 소형 FC + `tanh` 스케일링. $\rho$는 생성자 인자로 노출 |
| 반복 $T$회, 가중치 공유 여부 | `GCNModel` 생성자 인자 `iterations`(기본 3), `shared_weights`(기본 True) |
| raw 출력 = 반복별 코너 열 $(N, T+1, 4, 2)$ | `GCNModel.forward`의 반환값. 기존 method들의 "raw output은 후처리 전 형태"라는 관례를 그대로 따름 |
| 심층 감독 SmoothL1 (단계별 가중합) | `src/components/losses.py`에 신규 클래스 추가(가칭 `DeepSupervisedSmoothL1Loss`). `raw_output (N, T+1, 4, 2)`과 target `(N, 4, 2)`를 받아 단계별 SmoothL1 평균의 가중합을 계산. 가중치 스킴(균등/후기 강조)은 생성자 인자로 노출 |
| 후처리 (마지막 반복 + `[0,1]` 클램프) | `GCNPostprocessor.__call__`이 `raw_output[:, -1]`을 취해 클램프 |
| preprocessor | `GCNPreprocessor`는 표준 코너 `(N, 4, 2)`를 그대로 반환하는 passthrough(부록 A: 단계별 타깃 변환 불필요). `BasePreprocessor` 인터페이스 일관성을 위해 클래스는 유지 |
| 평가 지표 | 후처리된 최종 코너에 기존 `PolygonIoU` 재사용 |

## 확정 결정

- 디렉터리·파일 구성: `ridge`(0008)와 동일하게 `src/methods/gcn/{__init__.py,model.py,preprocessor.py,postprocessor.py,wrapper.py}` 5개 파일로 구성한다.
- head 문자열: `GCNWrapper`는 `head="gcn"`만 지원한다(다른 값은 `ValueError`). `head` 인자는
  아키텍처 선택이 아니라 검증용 상수로만 쓰인다(기존 method들과 동일 패턴).
- backbone 선택: `ridge`와 동일한 인코더 후보군(`custom`/`resnet18-50`/`efficientnet_b0`/
  `swin_t`/`vgg16-19`/timm CNN)을 그대로 지원한다. 0010 플랜이 먼저 적용되면 인자명은
  `network=`로, 아직 적용 전이면 `backbone=`으로 받는다(적용 순서는 후속 작업에서 확인).
- GCN 그래프 구조: 정점 4개, 간선은 4-순환(`(1,2),(2,3),(3,4),(4,1)`)으로 고정한다. 학습 가능한
  인접행렬이나 다른 토폴로지는 이 플랜의 범위에 포함하지 않는다.
- 반복 횟수 $T$와 GCN 계층 수 $L$: 각각 기본값 `iterations=3`, `num_layers=2`로 노출하되, 실제
  최적값 탐색(하이퍼파라미터 튜닝)은 이 플랜의 범위 밖이며 후속 실험에서 결정한다.
- 손실 함수: 새 공유 컴포넌트 `DeepSupervisedSmoothL1Loss`를 `src/components/losses.py`에 추가한다
  (기존 `HeatmapFocalLoss` 옆에 위치). 단계 가중치 기본값은 균등(`w_t = 1`)으로 하고, 후기 단계
  강조(`w_t ∝ t+1`) 옵션을 생성자 인자로 제공한다.
- optimizer/scheduler: `RidgeWrapper`와 동일한 2단계 warmup(backbone freeze → 낮은 lr로
  unfreeze) 패턴을 재사용한다. 방법론 문서 부록 B의 3단계 학습 전략(사전학습/도메인 적응/
  파인튜닝)은 데이터셋 구성·학습 캠페인 차원의 절차이며 wrapper 코드 구조로 표현할 대상이
  아니므로 이 플랜에서는 다루지 않는다(제외 항목 참조).
- factory 연결: `src/core/factory.py`의 `get_wrapper`에 `method == "gcn"` 분기를 추가하고
  `GCNWrapper`를 import한다. 0010 적용 여부에 따라 인자명이 `method`/`model` 중 무엇이 될지는
  후속 작업에서 그 시점의 최신 상태를 따른다.

## 구현 결과

이 플랜은 0010이 적용된 이후 상태에서 구현했다. 따라서 실제 경로는 `src/models/gcn/`이고
아키텍처 인자는 `network=`이며, factory와 스크립트는 `model="gcn"` 문자열로 dispatch한다. ver1
(`../260701_roi-corner-detection-ver1/src/models/gcn/`)의 구현을 참고하되, ver3의 컴포넌트 구조에
맞게 다음을 변경했다.

- backbone은 raw torchvision resnet 직접 사용 대신 `FeatureExtractor` + `CNNBackboneAdapter(keep_spatial=True, keep_stages=False)`로 통일했다. `custom`/`resnet`/`efficientnet`/`swin`/`vgg`/timm CNN을 지원하며 기본값은 `custom`이다.
- 초기 추정 head는 `src/models/gcn/model.py`의 로컬 `InitHead`(stride conv x2 + AdaptiveAvgPool + Flatten + Linear)로 구현했다.
- GCN 정제는 로컬 `GCNRefiner`로 구현하고, 4-순환 정규화 인접행렬은 `adjacency` 버퍼로 고정 등록했다. `iterations`, `num_layers`, `shared_weights`, `offset_radius`, `hidden_dim`을 `GCNModel` 생성자 인자로 노출했다.
- 심층 감독 손실은 ver1의 wrapper 내부 override 대신 공유 컴포넌트 `DeepSupervisedSmoothL1Loss`로 추출해 `src/components/losses.py`에 추가했다. 표준 `BaseWrapper.compute_losses`가 `raw_output (N, T+1, 4, 2)`와 passthrough target `(N, 4, 2)`를 그대로 전달하므로 wrapper override가 필요 없다. 단계 가중치는 균등(`late_emphasis=False`)과 후기 강조(`late_emphasis=True`)를 모두 지원한다.
- optimizer/scheduler는 `RidgeWrapper`와 동일한 2단계 warmup 패턴을 재사용했고, `scripts/config.py`의 `warmup_models`에 `gcn`을 추가했다.

## 범위

포함 항목(후속 코드 작업 대상):

- `src/methods/gcn/model.py`: `GCNModel` 신규 작성. backbone 선택 로직(`ridge`와 공유 가능한
  부분은 재사용), 공간 보존 초기 추정 head, 쌍선형 정점 샘플링, 고정 인접행렬 GCN 정제 모듈,
  $\rho\tanh$ offset head, $T$회 반복 루프를 포함한다. `forward`는 `(N, T+1, 4, 2)` 반복별
  코너 열을 반환한다.
- `src/methods/gcn/preprocessor.py`: `GCNPreprocessor` 신규 작성(passthrough, 부록 A 근거).
- `src/methods/gcn/postprocessor.py`: `GCNPostprocessor` 신규 작성. 마지막 반복 좌표를 취해
  `[0, 1]` 클램프한다.
- `src/methods/gcn/wrapper.py`: `GCNWrapper` 신규 작성. `GCNModel`/`GCNPreprocessor`/
  `GCNPostprocessor`를 구성하고, 신규 `DeepSupervisedSmoothL1Loss`와 기존 `PolygonIoU`를
  기본 손실/지표로 설정하며, `ridge`와 동일한 2단계 warmup optimizer/scheduler를 구현한다.
- `src/methods/gcn/__init__.py`: 빈 패키지 마커.
- `src/components/losses.py`: `DeepSupervisedSmoothL1Loss` 클래스를 추가한다.
- `src/core/factory.py`: `"gcn"` method 분기와 `GCNWrapper` import를 추가한다.
- `scripts/config.py`·`scripts/batch_config.py`: `"gcn"`을 지원 method(또는 0010 적용 후
  model) 목록과 `warmup_methods`(warmup이 필요하면)에 추가한다.

제외 항목:

- 방법론 문서 부록 B의 3단계 학습 전략(SmartDoc/MIDV-2020 사전학습 → 합성 fringe 도메인 적응 →
  실측 파인튜닝) 실행, 즉 실제 데이터셋 구성과 학습 캠페인은 이 플랜의 범위에 포함하지 않는다.
- 초기 추정 head를 `reg` method의 기존 head 컴포넌트와 통합하거나 공유 컴포넌트로 추출하는 작업은
  하지 않는다(중복 여부 확인은 후속 작업).
- GCN 그래프 토폴로지를 학습 가능하게 만들거나 정점 수를 4개 이외로 확장하는 것은 다루지 않는다.
- 다른 method(`reg`, `seg`, `det`, `torchseg`, `torchdet`, `heatmap`/`peak`, `linemap`/`ridge`,
  `yolo`, `detr`)의 코드는 변경하지 않는다.
- 0010(`method`→`model`/`network` 인자 통합), 0009(`heatmap`/`linemap` 개명)의 적용 순서 조율은
  이 플랜에서 결정하지 않는다. `gcn`은 두 플랜이 어느 순서로 적용되든 동일한 패턴(단일 backbone
  선택 인자, `head` 검증 상수)을 따르므로 충돌하지 않는다.
- 실제 학습 실행, 하이퍼파라미터($T$, $L$, $\rho$, 손실 가중치) 탐색과 `PolygonIoU` 수치 검증은
  이 플랜의 범위 밖이며 후속 작업에서 수행한다.

## 완료 기준

- `src/methods/gcn/`에 5개 파일이 위 범위대로 존재한다.
- `GCNModel.forward`가 `(N, T+1, 4, 2)` shape를 반환하고, `GCNPostprocessor`가 이를
  `(N, 4, 2)` 표준 코너로 변환한다.
- `GCNModel`, `GCNWrapper`가 `BaseModel`, `BaseWrapper` 인터페이스를 만족하여 기존 method들과
  동일한 방식으로 `Trainer`/`Evaluator`/`Predictor`에 연결될 수 있다.
- `src/components/losses.py`의 `DeepSupervisedSmoothL1Loss`가 단계별 SmoothL1 가중합을
  올바르게 계산한다(균등/후기 강조 두 가중치 스킴 모두 동작).
- `src/core/factory.py`가 `"gcn"` method(또는 model) 문자열로 `GCNWrapper`를 dispatch한다.
- 다른 method의 코드는 수정되지 않는다.

## 검증

구현 후 conda 환경 `pytorch_env`에서 다음을 확인했다.

- import 검증: `PYTHONPATH=<project-root> python -c "import src.core.factory; import src.models.gcn.wrapper"` 오류 없음.
- 단위 검증: `(2, 3, 224, 224)` 이미지 배치와 `(2, 4, 2)` 정답 코너로 `network`가 `custom`과 `resnet18`일 때, `shared_weights`가 True와 False일 때 모두 `GCNModel.forward`가 `(2, 4, 4, 2)`(즉 $T+1=4$) shape를 반환했다. `DeepSupervisedSmoothL1Loss`가 균등/후기 강조 두 스킴에서 스칼라 손실과 backward를 정상 계산했고, `GCNPostprocessor`가 `(2, 4, 2)` 코너를 `[0, 1]` 범위로 반환했다.
- 통합 검증: `get_wrapper(model="gcn", network="custom", head="gcn", warmup_epochs=1)`로 생성한 wrapper가 `on_fit_start`/`on_epoch_start`(backbone freeze 후 unfreeze) $\to$ `train_step` $\to$ `eval_step` $\to$ `predict_step`을 shape 오류 없이 수행했고 `predict_step`이 `(2, 4, 2)`를 반환했다.
- 기존 method 회귀: `SegModel`/`RegModel`의 `custom` network 빌드와 forward가 backbone 목록 추가 후에도 그대로 동작함을 확인했다.
- 실제 학습 스크립트 실행(`scripts/train.py --model gcn --network custom --head gcn`)과 `PolygonIoU` 수치, `iterations` 값별 정제 상한 분석은 이 플랜의 범위 밖이며 후속 작업에서 확인한다.
