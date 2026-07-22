# 0020 LabelMe parser plan

이 문서는 LabelMe 라이브러리로 레이블링한 image와 polygon JSON을 `gt_corners.csv`로 변환하는 parser를
`src/data`에 추가하고 `create_data.py` CLI에 연결하는 작업을 기록한다.

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-22 |
| 적용 범위 | `src/data/labelme.py`, `data/create_data.py` |
| 관련 문서 | [Dataset Format](../guides/01-dataset-format.md), [src layout](../architecture/03-src-layout.md) |

## 1. 목적과 배경

`data/create_synthetic.py`는 image와 LabelMe polygon JSON을 `data/synthetic/synthetic_0X` 하위 폴더에
생성한다. 기존 `src/data`의 raw parser는 `smartdoc`, `midv2020`, `images` 세 가지뿐이라 LabelMe 형식의
polygon 주석을 학습용 `gt_corners.csv`로 변환하는 경로가 없다. 이 작업은 LabelMe polygon을 다른 parser와
동일한 CSV schema로 변환하는 `labelme` parser를 추가한다.

## 2. 범위

포함 항목은 다음과 같다.

- `src/data/labelme.py`에 `create_data(data_dir, output_path)` 함수를 추가한다.
- `data_dir` 아래 하위 폴더까지 재귀 탐색하여 모든 `*.json`을 수집한다. `glob.glob`의 `**` recursive
  패턴을 사용한다.
- 각 JSON의 `roi` polygon 4점을 `imageWidth`와 `imageHeight`로 정규화하고 `order_corners`,
  `is_invalid_corners`로 정렬과 검증을 수행한 뒤 기존 CSV schema로 저장한다.
- `data/create_data.py` CLI의 `--dataset` 선택지에 `labelme`를 추가하고 dispatch를 연결한다.

제외 항목은 다음과 같다.

- `data/create_synthetic.py`의 생성 로직 변경은 하지 않는다.
- `src/data/dataset.py` 이후의 CSV load, split, transform 흐름은 변경하지 않는다.
- dataset-format 가이드 본문 확장은 하지 않는다. 개별 raw parser는 canonical 가이드가 열거하지 않는다.

## 3. 완료 기준

다음을 모두 충족하면 이 plan을 Done으로 본다.

- `src/data/labelme.py`의 `create_data`가 다른 parser와 동일한 signature와 CSV header를 사용한다.
- `data_dir` 하위 폴더의 JSON까지 재귀로 수집한다.
- `data/create_data.py`에서 `--dataset labelme`로 실행 가능하다.
- `data/synthetic`를 `data_dir`로 지정한 예시 실행이 `data/synthetic/gt_corners.csv`를 생성한다.

## 4. 검증

`pytorch_env`에서 다음을 실행하여 확인한다.

- `python -m py_compile src/data/labelme.py data/create_data.py`로 문법을 확인한다.
- 다음 command로 예시 CSV를 생성하고 행 수와 폴더 분포를 확인한다.

```bash
python data/create_data.py --dataset labelme \
  --data_dir data/synthetic \
  --output_path data/synthetic/gt_corners.csv
```

`data/synthetic`의 세 하위 폴더에서 각각 20개, 총 60개 데이터 행과 header 1행이 생성되면 통과로 본다.
