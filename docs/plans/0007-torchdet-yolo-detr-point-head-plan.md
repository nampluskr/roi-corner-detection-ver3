---
상태: Done
작성일: 2026-07-21
완료일: 2026-07-21
적용 범위: `src/methods/torchdet/wrapper.py`, `src/methods/yolo/wrapper.py`, `src/methods/detr/wrapper.py`, `scripts/batch_config.py`
관련 문서: [../README.md](../README.md), [../CLAUDE.md](../CLAUDE.md), [0005-methods-restructure-plan.md](0005-methods-restructure-plan.md)
---

## 목적과 배경

`det` method의 최근 학습 결과에서 `DetectionHead`의 `head` 값이 `box`일 때보다 `point`일 때
valid iou가 높게 나타났다. `custom_box`는 5 epoch 시점 iou 약 0.5933, `custom_point`는 약
0.6193으로, corner 위치를 4채널 box 대신 2채널 center offset으로 예측하는 편이 corner detection
과제에 더 유리하다는 근거가 관측되었다.

`DetModel`은 `head="box"`와 `head="point"`를 실제로 지원한다. `head` 값이 `src.components.heads`의
`BOX_CHANNELS = {"box": 4, "point": 2}`를 통해 `DetectionHead`의 regression 채널 수로 연결되기
때문이다. 반면 `TorchDetModel`, `YoloModel`, `DetrModel`은 각각 torchvision, Ultralytics, Hugging
Face의 whole-model detector를 그대로 사용하므로 regression head 구조를 바꿀 수 없다. 이 세 wrapper는
`head` kwarg를 CLI 호환 목적으로 받기만 하고 무시한다.

세 whole-model detector는 이미 corner를 고정 크기 pseudo-box로 표현하고, 추론 시 box의 center만
꺼내 corner로 사용한다. `TorchDetPreprocessor`, `YoloPreprocessor`, `DetrPreprocessor`가
`box_size`로 pseudo-box의 크기를 만들고, 대응하는 postprocessor는 box의 center만 복원한다. 따라서
`box_size`는 학습 target의 크기일 뿐이며 학습되거나 평가되는 값이 아니다. `box_size`를 작게 하면
center 감독이 지배적이 되어 point 방식에 가까워진다.

이 작업은 `DetModel`처럼 실제 keypoint head를 추가하지 않고, 세 wrapper가 `head` 값에 따라
`box_size`를 자동으로 선택하도록 하여 point 방식을 근사한다. box는 `box_size` 0.3, point는
`box_size` 0.1을 사용한다.

## 범위

포함 항목은 다음과 같다.

- `src/methods/torchdet/wrapper.py`, `src/methods/yolo/wrapper.py`,
  `src/methods/detr/wrapper.py` 세 wrapper에 동일한 패턴을 적용한다. 각 wrapper에 module-level
  `HEAD_BOX_SIZE = {"box": 0.3, "point": 0.1}`를 두고, 생성자에서 `box_size`가 명시되지 않으면
  `head` 값으로부터 `box_size`를 유도한다.
- 세 wrapper의 `box_size` 생성자 기본값을 `None`으로 바꾸어, 미지정 시 `HEAD_BOX_SIZE[head]`로
  해석되게 한다. 그 결과 `head="box"`는 세 모델 모두 `box_size` 0.3으로 통일된다.
- `head`가 `HEAD_BOX_SIZE`에 없으면 `DetModel` 및 `DetectionHead`의 오류 메시지 형식과 동일하게
  `ValueError`를 발생시킨다.
- `scripts/batch_config.py`의 torchdet, yolo, detr 예시 항목에 `head`가 `box`인 항목과 `point`인
  항목을 각각 둔다. `head` 필드는 이미 `scripts/config.py`의 `get_wrapper_kwargs`를 통해 wrapper로
  전달되며, 실험 이름에도 `head`가 포함된다.

제외 항목은 다음과 같다.

- 실제 keypoint 또는 pose head를 추가하는 방향(torchvision keypointrcnn, Ultralytics pose,
  DETR용 커스텀 keypoint head)은 포함하지 않는다.
- `DetectionHead`처럼 regression 채널 수를 바꾸는 변경은 세 whole-model detector에 적용할 수
  없으므로 포함하지 않는다.
- `--box_size`를 CLI 및 config 필드로 노출하지 않는다. `box_size`는 `head`로부터 자동 유도한다.
- `DetModel`(`det` method)의 기존 box/point 동작은 변경하지 않는다.
- 세 wrapper의 preprocessor와 postprocessor 로직 자체는 변경하지 않는다. `box_size`를 어떻게
  결정하는지만 wrapper에서 조정한다.
- `docs/README.md`는 이 프로젝트 재구성 workflow 자체의 최상위 색인이며 method별 구현 세부사항을
  담는 문서가 아니므로 수정하지 않는다. 이 plan 문서 자체가 이 변경의 canonical 근거다.

## 완료 기준

- `TorchDetWrapper`, `YoloWrapper`, `DetrWrapper`가 각각 `head="box"`일 때 `box_size` 0.3,
  `head="point"`일 때 `box_size` 0.1로 preprocessor를 구성한다.
- `box_size`를 명시적으로 전달하면 그 값이 `head` 기반 자동 유도보다 우선한다.
- 알 수 없는 `head` 값에 대해 세 wrapper 모두 명확한 `ValueError`를 발생시킨다.
- 세 method를 `head="box"`와 `head="point"`로 실행하면 서로 다른 실험 이름이 생성되며, 두 경우
  모두 발산 없이 학습된다.
- `DetModel`을 포함한 다른 method의 동작은 변경되지 않는다.

## 검증

검증은 프로젝트 규칙에 따라 `pytorch_env`에서 project root를 기준으로 수행한다.

```bash
conda activate pytorch_env
cd <project-root>
```

검증 항목은 다음과 같다.

- kwargs 확인: `TorchDetWrapper`(fasterrcnn_resnet50_fpn), `YoloWrapper`(yolov8n),
  `DetrWrapper`(detr_resnet50)를 `head="box"`와 `head="point"`로 각각 생성하여 preprocessor의
  box 크기가 0.3과 0.1로 해석되는지 확인했다. 세 wrapper 모두 `head="bogus"`에 대해 `ValueError`를
  발생시키는 것을 확인했다.
- 짧은 학습: `python scripts/train.py`로 세 method를 `head="point"`, batch_size 4, max_epochs 2,
  train_size 40으로 각각 실행했다. 세 경우 모두 loss가 유한하게 유지되고 epoch가 진행됨에 따라
  감소했으며, valid iou가 0보다 큰 값을 보였다(torchdet 0.002, yolo 0.038, detr 0.116; 표본이
  작은 smoke test이므로 절대값보다 발산하지 않고 학습이 진행됨을 확인하는 데 의미가 있다).
- `scripts/config.py`의 `get_experiment`로 동일 model에 대해 `head="box"`와 `head="point"`가
  서로 다른 실험 이름(`torchdet_bs4_ep5_fasterrcnn_resnet50_fpn_box` vs
  `torchdet_bs4_ep5_fasterrcnn_resnet50_fpn_point`)을 생성하는 것을 확인했다.
