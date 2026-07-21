---
상태: Draft
작성일: 2026-07-21
완료일: (미정)
적용 범위: ver3 `src/models/hybrid/model.py`, `src/models/hybrid/preprocessor.py`, `src/models/hybrid/postprocessor.py`, `src/models/hybrid/wrapper.py`, `src/models/hybrid/__init__.py`, `src/core/factory.py`(dispatch 추가), `scripts/config.py`·`scripts/batch_config.py`(model 목록 추가)
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0009-peak-ridge-naming-plan.md](0009-peak-ridge-naming-plan.md), [0010-method-to-model-and-network-arg-plan.md](0010-method-to-model-and-network-arg-plan.md), [0011-gcn-method-plan.md](0011-gcn-method-plan.md), [DL + Classical CV Hybrid 방법론 문서](https://github.com/nampluskr/roi-corner-detection/blob/main/docs/models/05_dl-classical-cv-hybrid.md)
---

## 목적과 배경

기존 종단간 학습 계열 method들(`reg`/`seg`/`heatmap`/`ridge`/`gcn` 등)은 코너 좌표 또는 그와
동치인 dense 표현을 딥러닝으로 직접 예측한다. `hybrid` model은 "대략 어디인가"(강건성)는
딥러닝이, "정확히 어디인가"(정밀성)는 고전 컴퓨터 비전이 담당하는 2단계 방법론이다.

1. **1단계(학습 대상)**: 경량 세그멘테이션 신경망(MobileNetV3 인코더 + U-Net 디코더)이 사각형
   영역의 이진 마스크 확률 지도를 예측한다. 이 단계에만 학습 파라미터가 있다.
2. **2단계(학습 없음, 결정적)**: 마스크를 이진화한 뒤 고전 CV 연쇄
   `Canny → HoughLinesP → 4변 그룹화·교점 → cornerSubPix`로 코너를 서브픽셀 정밀도로 계산한다.

이 설계의 이점은 (a) 학습 파라미터가 세그멘테이션에만 있어 모델이 작고 CPU 추론에 유리하며
(F6 최적 후보), (b) cornerSubPix가 격자 제약 없는 서브픽셀 정밀도를 제공하고(F5 상위 후보),
(c) 마스크/엣지/직선/교점 중간 산출물이 모두 시각화 가능해 실패 단계를 특정할 수 있다는 것이다.
대가는 후처리 연쇄가 길어 실패 지점이 다양하고, 비학습 파라미터(Canny 임계, Hough 투표 수,
cornerSubPix 창 크기 등)의 도메인 튜닝이 필요하다는 점이다.

`hybrid`의 학습 타깃(채워진 사각형 마스크)과 실패 시 fallback(마스크 컨투어 폴리곤 근사)은
기존 `seg` method와 동일하다. 따라서 `seg`의 마스크 preprocessor와 컨투어 후처리 로직을 최대한
재사용하는 것이 이 플랜의 핵심 재사용 전략이다.

상세 이론(depthwise separable convolution, MobileNetV3 SE/hard-swish, U-Net skip, BCE+Dice
손실, Canny/Hough/cornerSubPix 수식, 3단계 학습 전략, F1-F8 제약 대응)은 위 "관련 문서"의 DL +
Classical CV Hybrid 방법론 문서를 따른다. 이 플랜은 그 이론을 ver3 코드 구조에 매핑하는 설계를
확정한다.

## 전제: 디렉터리 경로

사용자 요청이 `src/models/hybrid`를 지정하므로, 이 플랜은 [0010](0010-method-to-model-and-network-arg-plan.md)의
`src/methods/` → `src/models/` 이동이 이미 적용된(또는 이 플랜과 함께 적용되는) 상태를 전제로
`src/models/hybrid/` 경로에 작성한다. 만약 hybrid 구현 시점에 0010이 아직 적용되지 않았다면
경로는 `src/methods/hybrid/`가 되며, wrapper/model의 아키텍처 인자도 `network=` 대신
`backbone=`을 쓴다. 그 외 설계는 경로·인자명과 무관하게 동일하다.

## 이론 요약과 코드 매핑

| 이론 요소 | 코드 매핑 |
|---|---|
| 마스크 타깃 $M \in \{0,1\}^{H\times W}$ 래스터화 (부록 A) | `seg`의 마스크 preprocessor(반평면 판정으로 볼록 사각형 채우기)와 동일 방식. `HybridPreprocessor`는 `seg` preprocessor를 재사용하거나 동일 로직으로 `(N, 1, H, W)` 마스크 타깃 생성 |
| 세그멘테이션 신경망 $f_\theta$ (MobileNetV3 + U-Net) | `HybridModel`. backbone은 MobileNetV3(경량성이 설계 목표)로 고정하거나 기존 backbone 선택 로직에서 mobilenet 계열을 기본값으로 사용. 디코더는 `src/components/decoders.py`의 `SegDecoder`(U-Net 스타일) 재사용, 최종 1채널 head |
| BCE + Dice 손실 ($\lambda = 1$) | `src/components/losses.py`에 신규 `BCEDiceLoss`(가칭) 추가, 또는 기존에 BCE/Dice 컴포넌트가 있으면 조합. 마스크 logits `(N, 1, H, W)`와 타깃 마스크를 입력 |
| 후처리 연쇄 $g = g_{subpix} \circ g_{intersect} \circ g_{hough} \circ g_{canny}$ | `HybridPostprocessor.__call__`이 raw logits `(N, 1, H, W)`를 받아 sigmoid+이진화 후 OpenCV(`cv2`)로 Canny → HoughLinesP → 4변 그룹화·교점 → cornerSubPix를 수행해 `(N, 4, 2)` 반환. 배치 원소별 반복(고전 CV는 배치 벡터화 대상이 아님) |
| cornerSubPix 그래디언트 입력(원본 그레이스케일) | postprocessor가 원본 이미지를 참조해야 함. `BasePostprocessor.__call__` 시그니처가 원본 이미지를 받는지 확인 필요(아래 "미확인 사항" 참조). 받지 못하면 이진 마스크 경계 그래디언트로 대체하거나 시그니처 확장 검토 |
| fallback (마스크 컨투어 폴리곤 근사, seg 방식) | 고전 CV 연쇄 실패 시 `seg` postprocessor의 컨투어 근사 로직을 재사용해 후퇴 |
| 성공 플래그 / SR 집계 | 후처리가 성공/실패 플래그를 반환하고 Evaluator가 SR(Success Rate)을 집계하는 기존 관례를 따른다(README 4절). `is_invalid_corners` 퇴화 판정은 기존 유틸 재사용 |
| 평가 지표 | 성공 표본에 대해 기존 `PolygonIoU` 재사용 |

## 확정 결정

- 디렉터리·파일 구성: 기존 method들과 동일하게 `src/models/hybrid/{__init__.py,model.py,preprocessor.py,postprocessor.py,wrapper.py}` 5개 파일로 구성한다.
- head 문자열: `HybridWrapper`는 `head="hybrid"`만 지원한다(다른 값은 `ValueError`). 기존 method들의 head 검증 패턴을 그대로 따른다.
- backbone(0010 적용 시 `network`): 경량성이 설계 목표이므로 기본값을 MobileNetV3 계열로 한다. 기존 backbone 선택 로직에 mobilenet 계열이 없다면 timm CNN 경로로 로드하거나 `src/components/backbones.py`에 추가하는 것을 후속 작업에서 검토한다(이 플랜에서는 기본 backbone을 mobilenet 계열로 지정하는 것까지만 확정).
- 디코더: `SegDecoder`(U-Net 스타일 additive/skip 업샘플링)를 재사용하고 최종 1채널 마스크 head를 둔다. `seg` model의 head 구조와 동일하게 맞춘다.
- 손실: BCE + Dice 가중합($\lambda=1$). 신규 `BCEDiceLoss`를 `src/components/losses.py`에 추가한다.
- 후처리 라이브러리: 고전 CV 연쇄는 OpenCV(`cv2`) 함수(`Canny`, `HoughLinesP`, `cornerSubPix`)로 구현한다. `cv2` 의존성이 프로젝트에 이미 있는지 확인하고 없으면 후속 작업에서 추가한다.
- 후처리 파라미터: Canny 임계(`t_low`/`t_high`), Hough 투표 임계·최소 선분 길이, cornerSubPix 창 크기(예 11x11)를 `HybridPostprocessor` 생성자 인자로 노출하고 기본값을 방법론 문서 기준으로 설정한다. 격자 탐색을 통한 최종 튜닝은 범위 밖(부록 B).
- fallback: 고전 CV 연쇄 실패 시 `seg` 방식 컨투어 폴리곤 근사로 후퇴하고, 그마저 실패하면 성공 플래그 False를 반환한다.
- optimizer/scheduler: 세그멘테이션 단계에만 학습이 있으므로 `seg`/`ridge` wrapper와 동일한 2단계 warmup(backbone freeze → 낮은 lr unfreeze) 패턴을 재사용한다.
- factory 연결: `src/core/factory.py`의 `get_wrapper`에 `"hybrid"` 분기와 `HybridWrapper` import를 추가한다.

## 범위

포함 항목(후속 코드 작업 대상):

- `src/models/hybrid/model.py`: `HybridModel` 신규 작성. MobileNetV3 인코더 + `SegDecoder` + 1채널 마스크 head. `forward`는 `(N, 1, H, W)` 마스크 logits를 반환한다(고전 CV는 model이 아니라 postprocessor에 위치).
- `src/models/hybrid/preprocessor.py`: `HybridPreprocessor` 신규 작성. 표준 코너 `(N, 4, 2)`를 채운 사각형 마스크 `(N, 1, H, W)`로 래스터화한다. `seg` preprocessor 재사용 또는 동일 로직.
- `src/models/hybrid/postprocessor.py`: `HybridPostprocessor` 신규 작성. sigmoid+이진화 → Canny → HoughLinesP → 4변 그룹화·교점 → cornerSubPix → `[0,1]` 정규화. 실패 시 seg 컨투어 fallback, 성공 플래그 반환.
- `src/models/hybrid/wrapper.py`: `HybridWrapper` 신규 작성. `HybridModel`/`HybridPreprocessor`/`HybridPostprocessor`를 구성하고, 신규 `BCEDiceLoss`와 기존 `PolygonIoU`를 기본 손실/지표로 설정하며, `seg`와 동일한 2단계 warmup optimizer/scheduler를 구현한다.
- `src/models/hybrid/__init__.py`: 빈 패키지 마커.
- `src/components/losses.py`: `BCEDiceLoss` 클래스 추가(기존 BCE/Dice 컴포넌트가 있으면 조합만).
- `src/core/factory.py`: `"hybrid"` 분기와 `HybridWrapper` import 추가.
- `scripts/config.py`·`scripts/batch_config.py`: `"hybrid"`를 지원 model 목록과(필요 시) `warmup_methods`에 추가.

제외 항목:

- 방법론 문서 부록 B의 3단계 학습 전략(공개 데이터 사전학습 → 합성 fringe 도메인 적응 → 실측 파인튜닝) 실행, 즉 실제 데이터셋 구성과 학습 캠페인은 범위에 포함하지 않는다.
- 후처리 비학습 파라미터(Canny 임계, Hough 투표 수/최소 선분 길이, cornerSubPix 창 크기)의 검증셋 격자 탐색과 도메인별 최종 확정은 범위 밖(부록 B).
- MobileNetV3 backbone을 `src/components/backbones.py`에 정식 추가할지, 기존 timm 경로로 로드할지의 결정은 후속 작업에서 확정한다(이 플랜은 기본 backbone을 mobilenet 계열로 지정하는 것까지만).
- `BasePostprocessor` 시그니처가 원본 이미지를 받지 못할 경우의 시그니처 확장은 별도 검토가 필요하며(아래 "미확인 사항"), 다른 method에 영향을 주는 변경이므로 이 플랜에서 확정하지 않는다.
- 다른 method/model(`reg`, `seg`, `det`, `torchseg`, `torchdet`, `heatmap`/`peak`, `linemap`/`ridge`, `gcn`, `yolo`, `detr`)의 코드는 변경하지 않는다.
- 실제 학습 실행과 후처리 파라미터 탐색, `PolygonIoU`/SR 수치 검증은 범위 밖이며 후속 작업에서 수행한다.

## 미확인 사항 (구현 전 확인 필요)

현재 로컬 리포에는 `src/methods/ridge/` 5개 파일만 존재하고, 이 플랜이 재사용을 전제한
`seg` method, `SegDecoder`(`src/components/decoders.py`), `BasePostprocessor`
(`src/methods/base/postprocessor.py`), `BaseWrapper`, `src/core/factory.py`,
`scripts/*`는 로컬에 아직 없다(플랜 문서상으로만 참조됨). 구현 착수 전 다음을 확인한다.

- `seg` preprocessor/postprocessor의 실제 클래스명·시그니처(마스크 래스터화, 컨투어 근사).
- `BasePostprocessor.__call__`이 원본 이미지에 접근 가능한지(cornerSubPix 그래디언트 입력에 필요).
- 후처리가 성공/실패 플래그를 반환하는 기존 관례와 Evaluator의 SR 집계 인터페이스.
- `src/components/losses.py`에 BCE/Dice 관련 기존 컴포넌트 존재 여부.
- `src/components/backbones.py`의 MobileNetV3 지원 여부, `cv2` 의존성 존재 여부.

## 완료 기준

- `src/models/hybrid/`에 5개 파일이 위 범위대로 존재한다.
- `HybridModel.forward`가 `(N, 1, H, W)` 마스크 logits를 반환하고, `HybridPostprocessor`가 고전 CV 연쇄로 `(N, 4, 2)` 코너를 계산하며 실패 시 seg 컨투어 fallback과 성공 플래그를 제공한다.
- `HybridModel`, `HybridWrapper`가 `BaseModel`, `BaseWrapper` 인터페이스를 만족하여 기존 method들과 동일한 방식으로 `Trainer`/`Evaluator`/`Predictor`에 연결될 수 있다.
- `src/components/losses.py`의 `BCEDiceLoss`가 BCE와 Dice 가중합($\lambda=1$)을 올바르게 계산한다.
- `src/core/factory.py`가 `"hybrid"` model 문자열로 `HybridWrapper`를 dispatch한다.
- 다른 method/model의 코드는 수정되지 않는다.

## 검증

이 플랜은 문서 작성만 수행하는 Draft 단계이며, 코드 변경과 검증은 아직 수행하지 않았다. 후속
작업에서 위 범위대로 구현한 뒤 다음을 이 섹션에 기록한다.

- import 검증: `PYTHONPATH=<project-root> python -c "import src.core.factory; import
  src.models.hybrid.wrapper"` 오류 없음.
- 단위 검증: 임의의 `(N, 3, 512, 512)` 이미지 배치와 `(N, 4, 2)` 정답 코너로 `HybridPreprocessor`
  마스크 생성, `HybridModel` forward, `BCEDiceLoss` 계산이 shape 오류 없이 동작하는지 확인.
- 후처리 검증: 정답 코너로 생성한 이상적 마스크를 `HybridPostprocessor`에 넣었을 때 고전 CV
  연쇄가 원래 코너를 (허용 오차 내) 복원하는지, 마스크가 파편화된 경우 fallback이 동작하고
  성공 플래그가 올바른지 확인.
- 실제 학습 스크립트 실행(`scripts/train.py --model hybrid --network mobilenetv3 --head hybrid`
  또는 0010 미적용 시 `--method hybrid`)과 `PolygonIoU`/SR 수치는 이 플랜 단계에서 수행하지
  않으며 후속 작업에서 확인한다.
