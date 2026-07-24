# reg model head 기본값 spatial 변경 계획

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `src/models/reg/model.py`, `src/models/reg/wrapper.py`, `scripts/config.py`, `docs/models/01-reg.md`, `docs/guides/02-cli-usage.md`, `README.md` |
| 관련 문서 | `docs/models/01-reg.md`, `docs/guides/02-cli-usage.md` |

## 목적과 배경

`reg` model의 `--head` 기본값은 현재 `gap`이다. corner 위치에는 spatial arrangement가 중요하고
`spatial` head가 이를 더 잘 보존하므로, 기본값을 `spatial`로 바꾼다. `--head`를 명시하지 않고
`reg` model을 실행하는 모든 경로(코드 default, CLI global default, 문서 예시)가 영향을 받는다.

## 범위

포함 항목은 다음과 같다.

- `src/models/reg/model.py`의 `RegModel.__init__` default를 `head="spatial"`로, `head = head or "gap"`
  fallback을 `head = head or "spatial"`로 변경한다.
- `src/models/reg/wrapper.py`의 `RegWrapper.__init__` default를 `head="spatial"`로 변경한다.
- `scripts/config.py`의 global parser default `head="gap"`을 `head="spatial"`로 변경한다.
- `docs/models/01-reg.md`의 커맨드 예시(`--head gap`)와 5절 설명에서 기본값 서술을 `spatial`로 갱신한다.
- `docs/guides/02-cli-usage.md`의 `--head` 기본값 표, 커맨드 예시, output path 예시(`reg_custom_gap_public`
  -> `reg_custom_spatial_public`)를 갱신한다.
- `README.md`의 `--head` 기본값 표와 커맨드 예시를 갱신한다.

제외 항목은 다음과 같다.

- `gap` head 자체의 구현, `GapHead`/`SpatialHead` 클래스는 변경하지 않는다.
- `reg` 외 다른 model의 head 기본값은 변경하지 않는다.
- 이미 생성된 `outputs/` 산출물 경로나 기존 checkpoint는 변경하지 않는다.

## 완료 기준

이 plan은 다음 조건을 만족하면 `Done`으로 볼 수 있다.

- `--head`를 지정하지 않고 `--model reg`를 실행하면 `spatial` head가 사용된다.
- `scripts/config.py`, `src/models/reg/model.py`, `src/models/reg/wrapper.py`의 default가 모두
  `spatial`로 일치한다.
- 위 세 문서의 `reg` 관련 커맨드 예시와 기본값 서술이 `spatial` 기준으로 일관되게 갱신된다.

## 검증

검증은 `pytorch_env`에서 수행한다.

- `RegModel()`, `RegWrapper()`를 인자 없이 생성해 `head_name`이 `spatial`인지 확인한다.
- `scripts/config.py`의 parser를 인자 없이 파싱해 `head` 기본값이 `spatial`인지 확인한다.
