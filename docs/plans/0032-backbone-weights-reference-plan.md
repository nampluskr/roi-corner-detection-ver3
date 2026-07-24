# Backbone 가중치 Reference 문서 추가 계획

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `docs/reference/03-backbones.md` 신설, `docs/README.md` 색인 갱신 |
| 관련 문서 | `docs/reference/01-losses.md`, `docs/reference/02-metrics.md`, `docs/architecture/02-model-assembly.md`, shared backbone catalog |

## 목적과 배경

`/mnt/d/backbones`에는 여러 workspace가 공유하는 pretrained weight 파일이 있다. shared backbone catalog는
이 경로의 46개 가중치 전체를 권장, 조건부, 비권장으로 분류한다. 현재 project는 더 적은 backbone과
external whole-model만 사용하므로, 현재 CLI가 실제로 참조하는 가중치만
정리한 별도 reference 문서가 필요하다.

## 범위

포함 항목은 다음과 같다.

- `docs/reference/03-backbones.md`를 신설한다. 현재 `reference/` 폴더 관례를 따라 번호를 이어서
  `03-backbones.md`로 명명한다.
- 문서 내용은 `src/components/backbones.py`, `src/models/torchseg/model.py`,
  `src/models/torchdet/model.py`, `src/models/yolo/model.py`, `src/models/detr/model.py`가 참조하는
  파일만 포함한다. shared backbone catalog의 조건부, 비권장 항목은 포함하지 않는다.
- 각 항목은 로컬 파일 경로와 byte 크기, architecture와 사전학습, 현재 project에서의 적용 방법(model, network,
  head 대응), 직접 URL, SHA-256을 담는다.
- `docs/README.md`의 reference 색인 표와 문서 구조 tree에 새 문서를 추가한다.

제외 항목은 다음과 같다.

- shared backbone catalog는 수정하지 않는다. 별도 workspace의 reference이므로 참조만 하고 변경하지 않는다.
- `/mnt/d/backbones`의 파일 다운로드나 검증 자동화 스크립트는 만들지 않는다.
- 현재 project에서 사용하지 않는 DINOv2, AlexNet, SqueezeNet, Wide-ResNet 계열 등 조건부/비권장 가중치는
  문서에 포함하지 않는다.

## 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `docs/reference/03-backbones.md`가 현재 project의 12개 model 중 pretrained weight를 사용하는 모든 backbone과
  external whole-model 가중치를 표로 포함한다.
- 표의 각 파일 크기와 SHA-256이 `/mnt/d/backbones`의 실제 파일과 일치한다.
- `docs/README.md`의 reference 순서 표와 문서 구조 tree가 새 문서를 반영한다.

## 검증

검증은 문서 생성과 대조로 수행하며 code 실행은 필요하지 않다.

- `sha256sum`과 `stat`로 표의 각 파일 크기, SHA-256과 실제 로컬 파일을 대조한다.
- `src/components/backbones.py`와 각 `src/models/*/model.py`의 weight 경로 dict에 있는 항목과 문서 표의
  항목 수가 일치하는지 확인한다.
