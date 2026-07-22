# 0021 data script rename plan

이 문서는 `data` 폴더의 세 스크립트를 산출물 대상이 드러나는 이름으로 변경하는 작업을 기록한다.

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-22 |
| 적용 범위 | `data/create_synthetic.py`, `data/create_data.py`, `data/fix_data.py` |
| 관련 문서 | [0020-labelme-parser-plan.md](0020-labelme-parser-plan.md) |

## 1. 목적과 배경

`data` 폴더의 세 스크립트는 `create_` 접두사를 공유하지만 성격이 다르다. `create_synthetic.py`는 합성
image와 LabelMe JSON을 새로 생성하는 image generator이고, `create_data.py`와 `fix_data.py`는 raw 주석을
`gt_corners.csv` 라벨로 변환하거나 정리하는 라벨 처리기다. 접두사가 같아 이 차이가 흐려지므로 산출물
대상을 파일명에 노출하여 성격을 명확히 한다.

## 2. 범위

포함 항목은 파일명 변경과 header 첫 줄 갱신이다.

| 현재 | 변경 |
| --- | --- |
| `data/create_synthetic.py` | `data/make_synthetic_images.py` |
| `data/create_data.py` | `data/make_gt_corners.py` |
| `data/fix_data.py` | `data/fix_gt_corners.py` |

각 파일의 첫 줄 `# path/from/project/root.py: ...` header를 새 경로에 맞게 갱신한다.

제외 항목은 다음과 같다.

- 스크립트 내부 로직 변경은 하지 않는다.
- `src/data`의 parser module 이름은 변경하지 않는다.
- 완료된 이력 문서인 `0020-labelme-parser-plan.md`의 본문은 수정하지 않는다.

## 3. 완료 기준

다음을 모두 충족하면 이 plan을 Done으로 본다.

- 세 파일이 새 이름으로 존재하고 이전 이름은 남지 않는다.
- 각 파일의 header 첫 줄이 새 경로를 가리킨다.
- 새 이름으로 CLI와 generator가 정상 실행된다.

## 4. 검증

`pytorch_env`에서 다음을 실행하여 확인한다.

- `python -m py_compile`로 세 파일의 문법을 확인한다.
- 다음 command로 라벨 CSV 생성이 정상 동작하는지 확인한다.

```bash
python data/make_gt_corners.py --dataset labelme \
  --data_dir data/synthetic \
  --output_path data/synthetic/gt_corners.csv
```

`data/synthetic`의 세 하위 폴더에서 총 60개 데이터 행과 header 1행이 생성되면 통과로 본다.
