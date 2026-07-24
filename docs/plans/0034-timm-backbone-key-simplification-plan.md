# timm Backbone 키 단순화 계획

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `src/components/backbones.py`의 `TimmBackbone` registry, `docs/reference/03-backbones.md`, `docs/models/01-reg.md`, `docs/plans/0010-method-to-model-and-network-arg-plan.md`, `scripts/batch_config.py` |
| 관련 문서 | `docs/reference/03-backbones.md`, `docs/plans/0032-backbone-weights-reference-plan.md`, `docs/plans/0033-mobilenet-efficientnet-variants-plan.md` |

## 목적과 배경

`TimmBackbone`이 사용하는 `--network` 값은 timm의 원본 dotted 식별자를 그대로 CLI 키로 사용한다.
`wide_resnet50_2.tv_in1k`, `deit_base_distilled_patch16_224.fb_in1k`, `cait_s24_224.fb_dist_in1k`는
길고 tag 규칙에 익숙하지 않으면 입력이 번거롭다. `timm.create_model()`은 정확한 원본 식별자 문자열이
필요하므로, CLI 키를 단순화하려면 alias와 원본 이름을 분리하는 mapping이 추가로 필요하다.

## 범위

포함 항목은 다음과 같다.

- `src/components/backbones.py`에 alias에서 원본 timm 식별자로 변환하는 `TIMM_MODEL_NAMES` dict를
  추가한다.

  - `wide_resnet50_2` -> `wide_resnet50_2.tv_in1k`
  - `deit_base_distilled` -> `deit_base_distilled_patch16_224.fb_in1k`
  - `cait_s24` -> `cait_s24_224.fb_dist_in1k`

- `TIMM_BACKBONE_WEIGHTS`, `TIMM_CNN_BACKBONES`, `TIMM_VIT_PREFIX_TOKENS`의 키를 위 alias로 교체한다.
  `TIMM_VIT_BACKBONES`, `SUPPORTED_TIMM_BACKBONES`는 파생값이므로 자동 반영된다.
- `TimmBackbone.__init__`이 `timm.create_model()` 호출 시 `TIMM_MODEL_NAMES[backbone]`으로 원본
  식별자를 전달하도록 수정한다. `backbone` 자체는 alias를 유지해 `self.backbone_name`, 오류 메시지,
  `TIMM_BACKBONE_WEIGHTS`/`TIMM_VIT_PREFIX_TOKENS` 조회에 그대로 사용한다.
- `docs/reference/03-backbones.md`의 timm 3개 행에서 alias를 `--network` 값으로 표기하고 timm ID는
  참고 정보로 유지한다.
- `docs/models/01-reg.md`, `docs/plans/0010-method-to-model-and-network-arg-plan.md`,
  `scripts/batch_config.py`의 예시 값을 alias로 갱신한다.

제외 항목은 다음과 같다.

- `TorchBackbone`의 `BACKBONE_WEIGHTS`/`BACKBONE_BUILDERS` 키는 이미 단순하므로 변경하지 않는다.
- `TIMM_BACKBONE_WEIGHTS`의 로컬 경로(`/mnt/d/backbones/<원본이름>/model.safetensors`)는 실제 파일
  경로이므로 alias로 변경하지 않는다.
- `src/models/*/model.py`의 backbone tuple 조합 로직은 변경하지 않는다. `TIMM_CNN_BACKBONES`,
  `TIMM_VIT_BACKBONES` tuple 값만 alias로 바뀌면 `--network` 옵션에 자동 반영된다.

## 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `TimmBackbone(backbone="wide_resnet50_2")`, `TimmBackbone(backbone="deit_base_distilled")`,
  `TimmBackbone(backbone="cait_s24")`가 정상 생성되고 `forward()` 결과가 원본 이름 기준과 동일한
  shape을 반환한다.
- 기존 dotted 이름으로 생성 시 `ValueError`가 발생한다(하위 호환 유지하지 않음).
- `docs/reference/03-backbones.md`, `docs/models/01-reg.md`,
  `docs/plans/0010-method-to-model-and-network-arg-plan.md`, `scripts/batch_config.py`가 alias를
  일관되게 사용한다.

## 검증

검증은 `pytorch_env`에서 수행한다.

- 3개 alias 각각으로 `TimmBackbone`을 생성하고 dummy image tensor로 `forward()`를 호출해 `stages`
  개수와 `final` shape이 리팩터링 전 원본 이름 기준과 동일한지 확인한다.
- `src/models/seg/model.py`, `src/models/hybrid/model.py` 등에서 alias가 `--network` 옵션으로
  노출되는지 import 확인으로 점검한다.
