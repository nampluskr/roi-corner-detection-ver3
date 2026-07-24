# MobileNet/EfficientNet Variant 추가 계획

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `src/components/backbones.py`의 torchvision backbone registry, `docs/reference/03-backbones.md` |
| 관련 문서 | `docs/reference/03-backbones.md`, `docs/plans/0032-backbone-weights-reference-plan.md` |

## 목적과 배경

composable backbone에는 현재 `efficientnet_b0`, `mobilenet_v3_large`만 등록되어 있다. `/mnt/d/backbones`에는
`mobilenet_v2`, `mobilenet_v3_small`, `efficientnet_b5` 가중치가 이미 존재하며, 이 variant를 추가하면
`det`, `gcn`, `hybrid`, `peak`, `ridge`, `seg` model에서 선택 가능한 backbone 폭이 넓어진다.

`efficientnet_b3`, `efficientnet_b7`은 `/mnt/d/backbones`에 파일이 없어 이번 범위에서 제외한다.
`efficientnet_b0_ra-3dd342df.pth`는 `efficientnet_b0`와 같은 architecture의 대체 가중치일 뿐 새 variant가
아니며, ImageNet 정확도 위주로 강화된 augmentation이 이 project의 fringe 패턴 기반 ROI 특성과 맞지 않을
수 있어 포함하지 않는다.

`TorchBackbone.__init__`의 `efficientnet`, `mobilenet` 분기는 family 단위로 `stage_indices`,
`stage_channels`, `stage_strides`를 한 번만 정의한다. 이 값은 `efficientnet_b0`, `mobilenet_v3_large`
각각의 `features` 구조에만 유효하므로, 다른 variant를 추가하려면 backbone 이름별로 값을 구분하는 lookup
구조가 필요하다. `forward()`의 stage 추출 로직 자체는 `stage_indices` 기반으로 이미 범용적이라 변경이
필요 없다.

## 범위

포함 항목은 다음과 같다.

- `BACKBONE_WEIGHTS`에 `mobilenet_v2`, `mobilenet_v3_small`, `efficientnet_b5` 가중치 경로를 추가한다.
- `BACKBONE_BUILDERS`에 `models.mobilenet_v2`, `models.mobilenet_v3_small`, `models.efficientnet_b5`
  생성자를 추가한다.
- `EFFICIENTNET_BACKBONES`에 `efficientnet_b5`를, `MOBILENET_BACKBONES`에 `mobilenet_v2`,
  `mobilenet_v3_small`을 추가한다.
- `TorchBackbone.__init__`의 `efficientnet`, `mobilenet` 분기를 family 단위 상수 대신 backbone 이름별
  `stage_indices`, `stage_channels`, `stage_strides` lookup dict(`EFFICIENTNET_STAGE_INFO`,
  `MOBILENET_STAGE_INFO`)로 교체한다. 기존 `efficientnet_b0`, `mobilenet_v3_large` 값도 이 dict로
  옮기되 수치는 그대로 유지한다.
- 각 신규 variant의 stage 값은 224 입력 기준 `features` forward trace로 확인한 stride 2, 4, 8, 16, 32
  구간의 마지막 등장 인덱스와 channel 수를 사용한다.

  - `mobilenet_v2`: `stage_indices=(1, 3, 6, 13, 18)`, `stage_channels=(16, 24, 32, 96, 1280)`
  - `mobilenet_v3_small`: `stage_indices=(0, 1, 3, 8, 12)`, `stage_channels=(16, 16, 24, 48, 576)`
  - `efficientnet_b5`: `stage_indices=(1, 2, 3, 5, 8)`, `stage_channels=(24, 40, 64, 176, 2048)`

  모든 variant의 `stage_strides`는 `(2, 4, 8, 16, 32)`다.
- `docs/reference/03-backbones.md`의 composable backbone 표에 3개 행을 추가한다.

제외 항목은 다음과 같다.

- `efficientnet_b3`, `efficientnet_b7`은 로컬 가중치 파일이 없어 추가하지 않는다.
- `efficientnet_b0_ra-3dd342df.pth`는 대체 가중치일 뿐 새 variant가 아니므로 추가하지 않는다.
- `TimmBackbone`, `src/models/*/model.py`의 `TORCH_*_BACKBONES` 조합 로직은 변경하지 않는다. 이미
  `EFFICIENTNET_BACKBONES`, `MOBILENET_BACKBONES` tuple을 그대로 참조해 조립하므로 tuple에 이름만
  추가되면 각 model의 `--network` 옵션에 자동으로 노출된다.
- `docs/architecture/02-model-assembly.md`는 model, network, head 축을 설명하는 문서이며 backbone별
  stage 세부값을 다루지 않으므로 변경하지 않는다.

## 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `mobilenet_v2`, `mobilenet_v3_small`, `efficientnet_b5`가 `SUPPORTED_BACKBONES`에 포함되고 각각
  `TorchBackbone(backbone=...)`으로 생성된다.
- 세 backbone 모두 `forward()`가 5개 stage feature map을 반환하고 마지막 stage의 channel 수가
  `out_channels`와 일치한다.
- 기존 `efficientnet_b0`, `mobilenet_v3_large`, `resnet18/34/50`, `vgg16/19`, `vit_b_16`, `swin_t`의
  동작과 반환값이 이전과 동일하다(회귀 없음).
- `docs/reference/03-backbones.md`의 표에 3개 신규 행이 추가되고 파일 크기, SHA-256이 실제 파일과
  일치한다.

## 검증

검증은 `pytorch_env`에서 수행한다.

- 신규 3개 backbone과 기존 9개 backbone 전체에 대해 `TorchBackbone(backbone=name, pretrained=True)`를
  생성하고 dummy image tensor로 `forward()`를 호출해 `stages` 개수와 `final` shape을 확인한다.
- `stage_channels[-1]`과 `out_channels`가 일치하는지 확인한다.
- `sha256sum`으로 3개 신규 가중치 파일의 hash가 문서 표와 일치하는지 확인한다.
