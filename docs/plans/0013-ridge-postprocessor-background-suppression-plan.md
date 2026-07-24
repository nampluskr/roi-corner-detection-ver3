---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/models/ridge/postprocessor.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0008-ridge-method-plan.md](0008-ridge-method-plan.md), [0009-peak-ridge-naming-plan.md](0009-peak-ridge-naming-plan.md)
---

## 목적과 배경

`ridge` model은 동일한 backbone, decoder, `FourChannelDenseHead`, `HeatmapFocalLoss`를 쓰는
`peak` model과 거의 모든 구성을 공유한다. 그럼에도 resnet18 backbone 기준 5 epoch 학습에서
`peak`은 IoU 0.86에 도달하는 반면 `ridge`는 IoU 0.015에 머문다.

원인을 분석한 결과 기하 설계 자체는 정상이었다. ground truth 코너를 preprocessor로 target을 만든 뒤
그대로 postprocessor에 통과시키는 oracle 왕복 테스트에서 코너 복원 오차는 0.002 미만이었다.

문제는 학습된 맵을 디코딩하는 방식에 있었다. 학습된 `ridge` 체크포인트의 출력은 최대 sigmoid 확률이
0.56, 평균 0.08로 능선과 배경의 대비가 약 7배에 불과한 흐릿한 맵이었다. `peak`의 postprocessor는
argmax를 쓰기 때문에 흐릿한 맵에서도 가장 밝은 픽셀이 대략 코너 위치에 있어 견딜 수 있다. 반면
`ridge`의 postprocessor는 sigmoid 확률을 가중치로 전체 픽셀에 대한 가중 평균 중심과 전역 PCA로 선을
적합한다. 배경 억제가 전혀 없어서 다수를 차지하는 약한 배경 확률이 중심과 공분산을 지배하고, 그
결과 가중 중심은 이미지 중앙으로, 공분산은 등방성으로 붕괴하여 인접 선의 교점이 퇴화한다. 실제
예측 코너 네 개 중 세 개가 (0.5, 0.5) 부근으로 모여 IoU가 0에 수렴했다.

같은 체크포인트에서 디코딩만 상대 임계값 기반 배경 억제로 바꾸면 재학습 없이 IoU가 0.013에서
상대 임계값 0.5일 때 0.60, 0.7일 때 0.70으로 회복되는 것을 실측으로 확인했다.

## 범위

포함 항목은 다음과 같다.

- `src/models/ridge/postprocessor.py`의 `RidgePostprocessor`에 채널별 상대 임계값 기반 배경 억제를
  추가한다. sigmoid 확률을 계산한 뒤 채널별 최대값의 일정 비율(`rel_thresh`) 미만인 픽셀의 가중치를
  0으로 만들고, 남은 능선 픽셀만으로 기존의 가중 중심과 PCA 선 적합을 수행한다.
- `rel_thresh`는 `__init__` 인자로 노출하고 기본값을 둔다. 기본값은 실측 결과와 sharp map에서의
  안정성을 함께 고려해 정한다.

제외 항목은 다음과 같다.

- preprocessor의 target 정의(무한 직선 대 유한 선분) 변경은 이 플랜에서 다루지 않는다. 별도 검토
  대상이다.
- `peak`, `seg` 등 다른 model의 postprocessor는 변경하지 않는다.
- 손실 함수, 학습 epoch 수, backbone 구성 등 학습 측 변경은 포함하지 않는다.
- 이미 저장된 체크포인트나 `outputs/` 산출물의 마이그레이션은 포함하지 않는다.

## 완료 기준

이 플랜이 `Done`으로 전환되기 위한 조건은 다음과 같다.

- `RidgePostprocessor`가 채널별 상대 임계값으로 배경을 억제한 뒤 코너를 복원한다.
- oracle 왕복 정확도가 유지된다(코너 오차 0.01 미만).
- 기존 `ridge` 체크포인트를 새 postprocessor로 재평가했을 때 IoU가 기존 대비 뚜렷하게 상승한다.

## 검증

conda `pytorch_env`에서 다음을 수행했다.

- oracle 왕복 테스트: `rel_thresh=0.5` 새 postprocessor로 sharp target 세 케이스의 코너 복원 최대
  오차가 0.0022로, 변경 전(0.0016)과 동등한 정확도를 유지했다.
- 기존 체크포인트
  `outputs/public/ridge/resnet18_ridge/ridge_bs4_ep5_resnet18_ridge/model.pth`를 새 postprocessor로
  test 1000장 재평가한 결과 IoU가 0.013에서 0.6611로 상승했다. 재학습 없이 디코딩만 바꾼 결과다.
- 참고로 임계값 실험에서 `rel_thresh` 0.5는 IoU 0.60, 0.7은 0.70이었다. sharp map에서의 안정성을
  고려해 기본값을 0.5로 두었다.
- 새 postprocessor를 반영한 end-to-end 재학습을 사용자가 수행했다. resnet18 backbone 5 epoch
  학습에서 best-weight(epoch 4) 검증 IoU가 0.688로, 변경 전 0.015 대비 뚜렷하게 상승했다.
- custom backbone은 이 변경만으로는 IoU가 회복되지 않았는데, 이는 postprocessor 문제가 아니라 맵
  자체가 형성되지 않은 underfitting 때문이다. 이 중 preprocessor의 고정 sigma 요인은
  [0014](0014-ridge-preprocessor-sigma-scaling-plan.md)에서 다룬다.
