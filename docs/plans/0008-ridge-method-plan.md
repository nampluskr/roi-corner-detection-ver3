---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: ver3 `src/methods/ridge/model.py`, `src/methods/ridge/preprocessor.py`, `src/methods/ridge/postprocessor.py`, `src/methods/ridge/wrapper.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md), [0006-heatmap-postprocessor-argmax-plan.md](0006-heatmap-postprocessor-argmax-plan.md), [0009-peak-ridge-naming-plan.md](0009-peak-ridge-naming-plan.md)
---

> 갱신 노트(2026-07-21): 이 method는 최초에 `linemap`이라는 이름으로 도입되었으나, 이후
> [0009](0009-peak-ridge-naming-plan.md)의 명명 확정에 따라 `ridge`로 개명되었다. 이 문서는 개명
> 결과를 반영하여 `ridge`/`Ridge*` 표기로 갱신되었다. `linemap`은 이 method의 옛 이름이다.

## 목적과 배경

기존 `heatmap` method(0009에서 `peak`으로 개명)는 각 코너를 점(point) 중심의 가우시안 피크로
표현한다. 이 표현을 산(mountain) 비유로 부를 때 `peak`이라 할 수 있으며, 대응하여 코너가 아니라
인접한 두 코너를 잇는 모서리(edge)를 산맥(mountain range, ridge)처럼 표현하는 새로운 방식을
`ridge`(도입 당시 이름 `linemap`)라는 별도 method로 도입한다.

`ridge`의 4개 채널은 4개의 모서리(TR-BR, BR-BL, BL-TL, TL-TR 등 인접 코너 쌍)에 대응한다. 각
채널의 dense map은 두 코너를 지나는 무한 직선을 능선(ridge)의 정상으로 삼아, 직선에서 수직 거리가
멀어질수록 값이 감소하는 형태를 가진다. 이때 능선은 두 코너를 잇는 선분이 아니라, 두 코너를 지나며
이미지 전체(위/아래 또는 좌/우 끝)까지 연장된 직선이어야 한다. 즉 채널별 dense map은 이미지 전체를
가로질러 절단하는 울타리 형태로 나타난다.

## 범위

포함 항목은 다음과 같다.

- `src/methods/ridge/model.py`: `RidgeModel`을 신규 작성한다. `heatmap`(→`peak`) method와 동일한
  백본(`custom`/`resnet`/`efficientnet`/`swin`/`vgg`/`timm-cnn`) 선택 로직, `FeatureExtractor`,
  `UNetDecoder`, 4채널 `HeatmapHead`를 재사용한다. 출력 채널 수(4)와 구조는 동일하므로 신규 head는
  만들지 않는다.
- `src/methods/ridge/preprocessor.py`: `RidgePreprocessor`를 신규 작성한다. `(N, 4, 2)` 정규화
  코너를 입력으로, 채널 `i`는 코너 `i`와 코너 `(i + 1) % 4`를 지나는 무한 직선까지의 수직 거리를
  가우시안으로 감쇠시켜 `(N, 4, H, W)` ridge 타겟을 만든다.
- `src/methods/ridge/postprocessor.py`: `RidgePostprocessor`를 신규 작성한다. 각 채널을
  sigmoid 확률로 가중한 점 구름으로 보고 가중 중심(centroid)과 주성분 방향(고유벡터)을 구해 능선의
  직선을 복원한다. 인접한 두 직선(`(i - 1) % 4`, `i`)의 교점을 코너 `i`로 계산하여 표준 `(N, 4, 2)`
  코너로 변환한다.
- `src/methods/ridge/wrapper.py`: `RidgeWrapper`를 신규 작성한다. `HeatmapWrapper`(→`PeakWrapper`)와
  동일한 2단계 warmup 학습 전략(backbone freeze 후 낮은 lr로 unfreeze), `HeatmapFocalLoss`,
  `PolygonIoU` 메트릭을 재사용한다.
- `src/methods/ridge/__init__.py`를 빈 패키지 마커로 추가한다.

제외 항목은 다음과 같다.

- 기존 `heatmap`(→`peak`) method의 파일(`model.py`, `preprocessor.py`, `postprocessor.py`,
  `wrapper.py`)은 변경하지 않는다. `heatmap` → `peak` 명칭 변경은 이 플랜의 범위에 포함하지 않으며
  [0009](0009-peak-ridge-naming-plan.md)에서 별도로 다룬다.
- `src/components/heads.py`, `src/components/decoders.py`, `src/components/losses.py`,
  `src/components/metrics.py`는 변경하지 않는다. 기존 `HeatmapHead`, `UNetDecoder`,
  `HeatmapFocalLoss`, `PolygonIoU`를 그대로 재사용한다.
- 다른 method(reg, seg, det, torchseg, torchdet, yolo, detr)는 변경하지 않는다.
- 데이터셋/라벨링 코드에서 코너 순서(TR/TL/BR/BL 등)를 확정하는 작업은 이 플랜의 범위에 포함하지
  않는다. `ridge`는 입력 코너 배열의 순서를 그대로 따른다.

## 완료 기준

- `RidgePreprocessor.__call__`이 `(N, 4, 2)` 코너를 받아 `(N, 4, H, W)` ridge 타겟을 반환하며,
  각 채널의 최댓값 능선이 두 코너를 지나는 무한 직선(선분이 아님)을 이룬다.
- `RidgePostprocessor.__call__`이 `(N, 4, H, W)` raw logits를 받아 인접 채널 쌍의 직선 교점으로
  `(N, 4, 2)` 정규화 코너를 복원한다.
- `RidgeModel`, `RidgeWrapper`가 `BaseModel`, `BaseWrapper`의 인터페이스를 그대로 만족하여
  기존 `heatmap`(→`peak`) method와 동일한 방식으로 `Trainer`/`Evaluator`/`Predictor`에 연결될 수
  있다.
- 기존 `heatmap`(→`peak`), 그 외 method의 코드는 수정되지 않는다.

## 검증

- 4개 파일(`model.py`, `preprocessor.py`, `postprocessor.py`, `wrapper.py`)을
  `src/methods/ridge/`에 생성했다(도입 당시 경로는 `src/methods/linemap/`이었고, 이후 `ridge`로
  개명하면서 클래스명 `Linemap*` → `Ridge*`, 속성 `linemap_stride` → `ridge_stride`, 인자
  `ridge_size` 등으로 함께 변경했다). 빈 `__init__.py`는 최초 도입 당시 누락되어 있었고,
  [0009](0009-peak-ridge-naming-plan.md)와 [0010](0010-method-to-model-and-network-arg-plan.md)을
  구현하는 작업에서 `src/models/ridge/__init__.py`로 추가했다.
- `heatmap`(→`peak`) method의 대응 파일과 구조를 대조하여 `BaseModel`/`BasePreprocessor`/
  `BasePostprocessor`/`BaseWrapper` 인터페이스를 동일하게 만족함을 확인했다.
- 실제 학습 스크립트(`scripts/train.py --method ridge`) 실행과 `PolygonIoU` 수치 검증은
  이 대화 세션에서 수행하지 않았다. 후속 작업에서 `HeatmapFocalLoss`가 ridge 타겟(양성 픽셀 비중이
  점 타겟보다 높음)에서도 유효한지, 그리고 postprocessor의 직선 교점 계산이 실제 학습된 모델
  출력에서 안정적으로 코너를 복원하는지 별도로 확인이 필요하다.
