---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/methods/heatmap/postprocessor.py`, `src/components/losses.py`, `src/methods/heatmap/wrapper.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md)
---

## 목적과 배경

`heatmap` method는 `scripts/batch_run.py` 실행 결과에서 mse loss가 0.115에서 0.001까지 안정적으로
수렴하지만, PolygonIoU는 0.008에서 0.06 수준에 머물러 다른 method(reg, seg, det) 대비 크게 낮게
나타난다.

최초 가설은 학습에 쓰는 loss(`sigmoid` + MSE)와 추론에 쓰는 postprocessor(softmax 기반
soft-argmax)의 목적함수 불일치였다. `HeatmapPostprocessor`를 soft-argmax에서 hard-argmax(sigmoid
후 최댓값 위치)로 바꾸어 재학습했으나 iou는 약 0.06에서 0.065 수준으로 거의 개선되지 않았다.
이 결과로 최초 가설은 기각한다.

실측 검증 결과 근본 원인은 heatmap target의 극심한 픽셀 클래스 불균형이다. `heatmap_size=112`
기준 채널당 전체 12544 픽셀 중 target > 0.1인 픽셀은 약 58개(약 0.5%)이며, 나머지 99% 이상은
거의 0이다. `HeatmapMSELoss`(`nn.MSELoss`)는 모든 픽셀에 동일한 가중치를 부여하므로, loss를
낮추는 가장 쉬운 방향은 전체 픽셀을 균일하게 낮은 값으로 만드는 것이다. 실제로 학습된 모델의
`sigmoid(raw_output)` 채널별 최댓값은 대부분 0.3 전후로, 뚜렷한 피크를 형성하지 못했다. 5 epoch
안에 배경 dominant loss에 소수 양성 픽셀의 gradient 기여가 묻혀 코너 위치를 학습하지 못하는
것이 iou 저하의 실제 원인이다.

## 범위

포함 항목은 다음과 같다.

- `src/methods/heatmap/postprocessor.py`의 `HeatmapPostprocessor`를 soft-argmax에서
  hard-argmax(sigmoid 후 최댓값 위치)로 변경한다. (완료, 유지)
- `src/components/losses.py`에 CenterNet 스타일 penalty-reduced pixelwise focal loss
  (`HeatmapFocalLoss`)를 추가한다. 배경 픽셀의 loss 기여를 target 값에 따라 감쇠시켜 소수 양성
  픽셀의 gradient 비중을 높인다.
- `src/methods/heatmap/wrapper.py`의 `HeatmapWrapper`에서 기본 loss를 `HeatmapMSELoss`에서
  `HeatmapFocalLoss`로 교체한다.
- 변경 사항에 대해 `pytorch_env`에서 import 검증과 `scripts/train.py --method heatmap` 재실행을
  통해 iou 개선 여부를 확인한다.

제외 항목은 다음과 같다.

- `HeatmapPreprocessor`, `HeatmapModel`은 변경하지 않는다. preprocessor의 target 생성 로직은
  실측으로 정확함을 확인했다.
- `HeatmapMSELoss` 클래스 자체는 삭제하지 않고 유지한다. 다른 곳에서 참조할 가능성을 배제하지
  않기 위함이며, heatmap wrapper의 기본값에서만 제외한다.
- 다른 heatmap 관련 구현은 수정하지 않는다.
- 다른 method(reg, seg, det, torchseg, torchdet, yolo, detr)의 loss, postprocessor, wrapper는
  변경하지 않는다.

## 완료 기준

- `HeatmapPostprocessor`가 sigmoid 적용 후 각 채널의 최댓값 위치(정수 픽셀 좌표를 정규화 좌표로
  변환)를 반환한다.
- `HeatmapFocalLoss`가 배경 픽셀 비중이 높은 heatmap target에서도 양성 픽셀 근방에 더 큰 가중치를
  부여하도록 구현된다.
- `HeatmapWrapper`의 기본 loss가 `HeatmapFocalLoss`로 설정된다.
- `heatmap` method의 `PolygonIoU`가 기존 결과(약 0.06) 대비 유의미하게 개선된다.
- 다른 method(reg, seg, det, torchseg, torchdet, yolo, detr)의 loss, postprocessor, wrapper는
  변경하지 않는다.

## 검증

- `PYTHONPATH=<project-root> python -c "import src.components.losses; import
  src.methods.heatmap.postprocessor; import src.methods.heatmap.wrapper"`로 import 오류가 없는지
  확인했다.
- `conda activate pytorch_env` 후 project root에서 `python scripts/train.py --method heatmap
  --batch_size 4 --max_epochs 5 --backbone custom --head heatmap --save`를 재실행하여 iou 추이를
  변경 전과 비교했다. 5 epoch 종료 시점 valid iou가 약 0.06(원래 sigmoid+MSE)에서 약 0.53(focal
  loss + hard-argmax)으로 개선되었으며, 두 차례 재실행에서 0.42, 0.53으로 재현성을 확인했다. epoch가
  진행될수록 iou가 계속 상승하는 추세였다(1 epoch 0.05, 3 epoch 0.11-0.17, 5 epoch 0.42-0.53).
