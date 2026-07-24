---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/models/ridge/preprocessor.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0013-ridge-postprocessor-background-suppression-plan.md](0013-ridge-postprocessor-background-suppression-plan.md)
---

## 목적과 배경

[0013](0013-ridge-postprocessor-background-suppression-plan.md)에서 postprocessor의 배경 억제를
추가한 뒤 resnet18 backbone은 IoU 0.69에 도달했다. 그러나 custom backbone은 같은 5 epoch 학습에서
IoU 0.017에 머문다. 원인 분석 결과 이는 메트릭 오류가 아니라 ridge 맵이 형성되지 않은
underfitting이었다. custom 체크포인트의 출력 맵은 최대 sigmoid 확률 0.141, 채널 최대/평균 대비 약
1.6배로 거의 평평했고, 능선이 없으니 중심 및 PCA 기반 디코딩이 이미지 중앙으로 붕괴했다.

underfitting에는 두 요인이 겹친다. 첫째, custom backbone은 사전학습 없이 처음부터 학습하므로 5
epoch으로는 부족하다. 이는 학습 설정으로 대응할 문제이며 이 플랜의 범위가 아니다. 둘째,
preprocessor의 Gaussian 폭 `sigma`가 맵 해상도와 무관하게 2.0px로 고정되어 있다. custom은 stride 2로
112x112 맵을 쓰는데, resnet18은 stride 4로 56x56 맵을 쓴다. 같은 2px sigma라도 112 맵에서는 능선이
맵 폭의 1.8%로, 56 맵의 3.6%에 비해 상대적으로 절반 두께다. 결과적으로 positive 픽셀이 더 희소해
focal loss가 배경에 지배되고 능선 형성이 더 어렵다.

이 플랜은 두 번째 요인만 다룬다. `sigma`를 맵 해상도에 비례하도록 스케일하여 stride가 달라도 능선의
상대 두께가 일정하게 유지되게 한다.

## 범위

포함 항목은 다음과 같다.

- `src/models/ridge/preprocessor.py`의 `RidgePreprocessor`에서 `sigma`가 명시되지 않으면
  `ridge_size`에 비례해 자동 결정되도록 한다. 기준은 `ridge_size = 56`에서 기존 값 2.0과 일치하도록
  정한다(즉 `sigma = ridge_size / 28.0`). 명시적으로 전달된 `sigma`는 그대로 존중한다.

제외 항목은 다음과 같다.

- 학습 epoch 수, warmup 설정, backbone 구성 등 학습 측 변경은 포함하지 않는다. custom의 from-scratch
  underfitting 자체는 별도로 학습 설정으로 대응한다.
- postprocessor는 변경하지 않는다(0013에서 완료).
- `peak` 등 다른 model의 preprocessor는 변경하지 않는다. peak은 argmax 디코딩이라 능선 두께에 둔감하다.
- 기존 체크포인트나 `outputs/` 산출물의 마이그레이션은 포함하지 않는다.

## 완료 기준

이 플랜이 `Done`으로 전환되기 위한 조건은 다음과 같다.

- `RidgePreprocessor`가 `sigma` 미지정 시 `ridge_size`에 비례해 폭을 정한다.
- `ridge_size = 56`에서 `sigma`가 정확히 2.0으로, 기존 resnet18 동작이 회귀 없이 유지된다.
- `ridge_size = 112`(custom)에서 `sigma`가 4.0으로 커져 능선의 상대 두께가 56 맵과 같아진다.
- oracle 왕복 정확도가 유지된다(코너 오차 0.01 미만).

## 검증

conda `pytorch_env`에서 다음을 수행했다.

- `RidgePreprocessor(56).sigma`는 2.0, `RidgePreprocessor(112).sigma`는 4.0으로 확인했다.
  resnet18(56 맵) 동작은 회귀 없이 유지되고 custom(112 맵)은 능선의 상대 두께가 56 맵과 같아졌다.
- 명시적 override `RidgePreprocessor(112, sigma=2.0).sigma`는 2.0으로 존중됨을 확인했다.
- oracle 왕복 코너 복원 최대 오차는 56 맵 0.0022, 112 맵 0.0014로 모두 0.01 미만이다.
- custom backbone end-to-end 재학습과 IoU 개선 확인은 사용자가 직접 수행한다.
