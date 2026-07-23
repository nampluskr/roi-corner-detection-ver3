# 0026: ridge head `ridge` -> `pcaline` 이름 변경

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `ridge` model의 head 이름 변경, 관련 문서와 슬라이드 자산 |
| 관련 문서 | `docs/plans/0023-ridge-peak-head-plan.md`, `docs/models/08-offset.md`(무관), `README.md`, `docs/models/03-dense-prediction.md`, `docs/guides/02-cli-usage.md`, `slides/outline.md`, `slides/README.md` |

## 목적과 배경

`ridge` model은 `--head`로 `ridge`와 `peakprod` 두 가지를 지원한다. head 이름 `ridge`가 model 이름
`ridge`와 동일해 `--model ridge --head ridge`처럼 같은 단어가 반복되어 구분이 어렵다. head가 실제로
수행하는 알고리즘인 PCA line fitting과 intersection을 드러내는 이름 `pcaline`으로 변경한다.

## 범위

포함 항목은 다음과 같다.

- `src/models/ridge/wrapper.py`의 기본 head 값과 유효성 검사 문자열
- `README.md`의 model registry 표 head 열
- `docs/guides/02-cli-usage.md`, `docs/models/03-dense-prediction.md`의 `--head ridge` 예시와 서술
- `scripts/batch_config.py`의 주석 처리된 예시 config
- `slides/outline.md`, `slides/README.md`의 head 서술과 이미지 참조
- `slides/assets/make_postprocess_figs.py`의 함수명, 저장 파일명, subplot 제목
- `slides/assets/postprocess_ridge.png` 파일명을 `postprocess_ridge_pcaline.png`로 변경

제외 항목은 다음과 같다.

- `src/models/ridge/postprocessor.py`의 class 이름(`RidgePostprocessor`)은 변경하지 않는다. head 문자열
  변경 범위 밖이며 model 이름을 딴 class 이름은 여전히 유효하다.
- `src/components/heads.py`는 조사 결과 head 이름 문자열을 갖지 않는 공용 `FourChannelDenseHead`만
  포함하므로 실질적인 변경이 없다.

## 완료 기준

- `--head pcaline`으로 `ridge` model이 정상 동작하고 `--head ridge`는 더 이상 유효한 값이 아니다.
- 위 범위의 모든 문서와 슬라이드 자산에서 `ridge` head 이름이 `pcaline`으로 일관되게 표기된다.
- 이미지 파일명과 참조가 일치한다.

## 검증

`PYTHONPATH`에 ver3 project root를 포함해 `RidgeWrapper(head="pcaline")` 생성과
`RidgeWrapper(head="ridge")`가 `ValueError`를 내는지 확인한다. 문서 변경은 grep으로 잔여
`--head ridge` 문자열이 없는지 확인한다.
