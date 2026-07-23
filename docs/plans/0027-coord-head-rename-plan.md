# 0027: `CoordGapHead` -> `GapHead`, `CoordSpatialHead` -> `SpatialHead` 이름 변경

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `src/components/heads.py`의 coordinate head class 이름과 사용처, 관련 canonical 문서 |
| 관련 문서 | `docs/architecture/02-model-assembly.md`, `docs/models/01-reg.md` |

## 목적과 배경

`Coord` 접두어는 `src/components/heads.py` 안에서 coordinate 계열 head를 detection이나 dense head와
구분하려는 의도였지만, 실제로는 `reg`와 `offset` model에서만 사용되고 다른 head class는 이미
`DetectionHead`, `FourChannelDenseHead`, `MaskHead`처럼 접두어 없이 역할만으로 명명되어 있어 일관성이
없었다. `GapHead`, `SpatialHead`로 줄여 다른 head class와 명명 규칙을 통일한다.

## 범위

포함 항목은 다음과 같다.

- `src/components/heads.py`의 class 정의
- `src/models/reg/model.py`, `src/models/offset/model.py`의 import와 사용
- `docs/architecture/02-model-assembly.md`의 조립 예시
- `docs/models/01-reg.md`의 head 설명

제외 항목은 다음과 같다.

- `docs/plans/0004-components-flatten-plan.md`, `docs/plans/0022-offset-model-plan.md`는 완료된 이력
  문서이므로 옛 이름을 그대로 보존한다.

## 완료 기준

- `GapHead`, `SpatialHead`로 `RegModel`과 `OffsetModel`이 정상 동작한다.
- 위 범위의 모든 문서에서 옛 class 이름이 남아 있지 않다.

## 검증

`PYTHONPATH`에 ver3 project root를 포함해 `RegModel(head="gap")`과 `OffsetModel(head="spatial")`을
생성하고 forward pass의 출력 shape를 확인했다. 문서 변경은 grep으로 활성 문서에 잔여 `CoordGapHead`,
`CoordSpatialHead` 문자열이 없는지 확인했다.
