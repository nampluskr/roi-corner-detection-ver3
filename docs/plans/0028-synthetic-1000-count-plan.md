# 0028 Synthetic 100-count Regeneration Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `data/make_synthetic_images.py`, `data/synthetic/`, `docs/guides/05-synthetic-generation.md` |
| 관련 문서 | [05-synthetic-generation.md](../guides/05-synthetic-generation.md), [0021-data-script-rename-plan.md](0021-data-script-rename-plan.md) |

## 1. 목적과 배경

현재 `data/synthetic/synthetic_01`, `synthetic_02`, `synthetic_03`은 각각 preview geometry profile
`preview_01`, `preview_02`, `preview_03` 조건으로 20장씩 생성된 표본이다. 세 조건 모두 표본 수를
100장으로 늘려 이후 학습에 사용할 합성 dataset의 규모를 확장한다. 기존 20장 폴더는 새 100장 dataset으로
대체하므로 삭제한다.

## 2. 범위

포함하는 작업은 다음과 같다.

- `data/make_synthetic_images.py`의 출력 폴더 이름 규칙을 `synthetic_0X` 고정 이름에서 `synthetic_0X_<count>`
  형태로 바꾼다. `PROFILE_TARGETS`는 base 이름만 유지하고, `generate_dataset`에서 `args.count`를 이름에
  덧붙인다. `preview_spec`과 `validate_output`의 표본 분포 로직은 이미 `count != 20`일 때 generic 비율
  기반 분기를 사용하므로 변경하지 않는다.
- 기존 `data/synthetic/synthetic_01`, `synthetic_02`, `synthetic_03` 폴더와 `data/synthetic/gt_corners.csv`를
  삭제한다.
- `python data/make_synthetic_images.py --count 100 ...`을 실행하여 `synthetic_01_100`, `synthetic_02_100`,
  `synthetic_03_100` 세 폴더를 새로 생성한다. `--seed`는 기존 문서 예시와 동일하게 `42`를 사용한다.
  `--background-dir`는 사용자가 지정하는 실제 경로를 그대로 사용한다.
- `python data/make_gt_corners.py --dataset labelme --data_dir data/synthetic --output_path
  data/synthetic/gt_corners.csv`를 실행하여 세 폴더의 JSON을 다시 `gt_corners.csv`로 변환한다.
- `docs/guides/05-synthetic-generation.md`의 출력 폴더 표, 실행 예시, `data/synthetic` 하위 폴더 설명을
  새 이름 규칙과 100장 예시에 맞게 갱신한다.

제외하는 작업은 다음과 같다.

- geometry profile별 조건 분포(`preview_spec`, `PREVIEW_03_*` 상수 등)의 값 자체는 바꾸지 않는다.
- `public`, `measured` 단계의 data나 문서는 바꾸지 않는다.
- `src/` 아래 학습 코드, dataset loader, split 로직은 바꾸지 않는다.

## 3. 완료 기준

- `data/synthetic/synthetic_01_100`, `synthetic_02_100`, `synthetic_03_100` 세 폴더가 각각 100쌍의
  PNG와 JSON을 담고 있다.
- 기존 `data/synthetic/synthetic_01`, `synthetic_02`, `synthetic_03` 폴더가 더 이상 존재하지 않는다.
- `data/synthetic/gt_corners.csv`가 세 폴더의 표본을 모두 반영해 재생성되어 있다.
- `docs/guides/05-synthetic-generation.md`의 출력 폴더 표와 실행 예시가 새 이름 규칙과 일치한다.

## 4. 검증

- `data/make_synthetic_images.py` 실행 로그에서 세 폴더 각각 `validated_pairs=100`을 확인한다.
- `data/make_gt_corners.py` 실행 후 `data/synthetic/gt_corners.csv`의 행 수가 300행(header 제외)인지
  확인한다.
- `conda activate pytorch_env` 환경에서 실행하며, project root에서 실행한다.
