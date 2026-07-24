# 0036 Measured Image Augmentation Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Approved |
| 작성일 | 2026-07-24 |
| 적용 범위 | `data/make_augmented_images.py`, `data/augmented/` |
| 관련 문서 | [05-data-strategy.md](../architecture/05-data-strategy.md), [05-synthetic-generation.md](../guides/05-synthetic-generation.md) |

## 1. 목적과 배경

실측 fringe data는 google 10장, H8 17장, Q8 16장, oppo 7장으로 표본 수가 적다. 각 장비의 실제 grayscale
TIFF만을 기반으로 제한된 기하 및 광학 변형을 적용해 원본 장비별 특성을 보존한 augmented image를 생성한다.
원본 `labels.csv`의 좌표는 입력으로 사용하지 않는다. 생성 image의 ROI는 image feature 기반으로 자동 검출하고,
LabelMe polygon annotation으로 별도 저장한다. 이 작업은 실제 장비 data를 대체하지 않으며, 소량 measured data의
offline pre-augmentation으로 사용한다.

## 2. 분석 결과

분석한 input data의 공통 특성과 case별 표본 수는 다음과 같다.

| case | 원본 수 | image 형식 | 해상도 | 평균 밝기 | 평균 표준편차 |
| --- | ---: | --- | --- | ---: | ---: |
| google | 10 | grayscale TIFF | 7920 x 6004 | 44.09 | 62.81 |
| h8 | 17 | grayscale TIFF | 7920 x 6004 | 39.22 | 59.09 |
| q8 | 16 | grayscale TIFF | 7920 x 6004 | 47.53 | 64.70 |
| oppo | 7 | grayscale TIFF | 7920 x 6004 | 37.15 | 56.62 |

각 case에는 source TIFF와 별도의 `labels.csv`가 있지만, 이 작업에서는 source TIFF만 분석한다. 모든 source는
`7920 x 6004` grayscale TIFF이며, 생성 image는 `1920 x 1080` grayscale TIFF로 축소한다. ROI는 생성된 image의
edge와 line feature에서 추정한 convex quadrilateral로 정의한다.

## 3. 범위

포함하는 작업은 다음과 같다.

- 단일 generator `data/make_augmented_images.py`를 추가한다. case별 별도 script는 만들지 않는다.
- `E:\\fringe_data\\training_all` 아래 google, H8, Q8, oppo source folder의 TIFF만 읽고 `labels.csv`는 읽지 않는다.
- source image를 순환 선택하고, 작은 translation, rotation, scale, perspective 변형을 적용한다.
- 상세 변형은 `data/make_synthetic_images.py`의 방식과 범위를 참고한다. case별 원본의 밝기와 대비 분포를
  기준으로 gain, bias, gamma, blur, sensor noise, stage illumination, 약한 stripe 또는 fringe variation을
  적용한다. 변형 범위는 ROI와 fringe 구조를 유지하는 보수적 값으로 제한한다.
- 기본 출력 해상도를 기존 synthetic data와 같은 `1920 x 1080`으로 설정하고, `--width`, `--height` option으로
  변경할 수 있게 한다. 학습용 `224` 또는 `512` resize는 data loader에서 수행한다.
- 생성된 image에서 edge, line, quadrilateral geometry를 이용해 ROI를 자동 검출한다. 검출한 polygon은 좌상단,
  우상단, 우하단, 좌하단 순서로 정렬한다.
- 첫 실행에서는 `--count 10`으로 `data/augmented/google_10`, `data/augmented/oppo_10`,
  `data/augmented/h8_10`, `data/augmented/q8_10`에 각 10장 TIFF와 대응 LabelMe JSON을 생성한다.
- 검토용 10장 output의 image와 annotation에 대해 사용자의 승인을 받은 후에만 `--count 100`을 실행한다.
- 승인 후 `--count 100`의 output path는 `data/augmented/google_100`, `data/augmented/h8_100`,
  `data/augmented/q8_100`, `data/augmented/oppo_100`으로 고정한다.
- 기존 output folder가 비어 있지 않으면 실패해 기존 data를 덮어쓰지 않는다.
- 생성 결과의 image 수와 JSON 수, image 크기, grayscale mode, LabelMe schema, 4점 polygon의 convexity와
  image 경계 내 좌표를 검증한다. 자동 검출의 confidence가 기준 미만이면 output을 저장하지 않고 실패한다.

제외하는 작업은 다음과 같다.

- `E:\\fringe_data\\training_all`의 원본 image와 `labels.csv`를 수정하지 않는다.
- 원본 `labels.csv`의 좌표를 생성, 변형 또는 annotation에 사용하지 않는다.
- synthetic generator와 기존 `data/synthetic/`의 data를 변경하지 않는다.
- 생성 data를 이용한 model training, CSV format 변환, canonical guide 변경은 포함하지 않는다.

## 4. 완료 기준

- `data/make_augmented_images.py --count 10`이 네 case의 검토용 output을 생성하고 검증을 통과한다.
- `data/augmented/google_10`, `data/augmented/oppo_10`, `data/augmented/h8_10`, `data/augmented/q8_10`의
  각 output folder에 10개 TIFF와 그에 대응하는 10개 LabelMe JSON이 있다.
- 검토용 output의 사용자 승인을 받은 후 `data/make_augmented_images.py --count 100`이 네 case의 output을
  생성하고 검증을 통과한다.
- `data/augmented/google_100`, `data/augmented/h8_100`, `data/augmented/q8_100`, `data/augmented/oppo_100`의
  각 output folder에 100개 TIFF와 그에 대응하는 100개 LabelMe JSON이 있다.
- 모든 기본 output image는 1920 x 1080 grayscale TIFF이고, 각 LabelMe annotation은 image 범위 안의 convex한
  4점 `roi` polygon을 가진다.
- 같은 seed와 input으로 실행하면 동일한 output이 생성된다.

## 5. 검증

WinPython 인터프리터 `C:\\winpython\\WPy64-31180\\python-3.11.8.amd64\\python.exe`로 다음 검증을 수행한다.

```powershell
& C:\\winpython\\WPy64-31180\\python-3.11.8.amd64\\python.exe data\\make_augmented_images.py --count 10
```

실행 로그의 case별 검증 완료 메시지와 TIFF 및 LabelMe JSON 수를 확인하고, 사용자가 검토용 output을
승인한 뒤에만 `--count 100` 실행을 허용한다. 출력 해상도를 1920 x 1080으로 축소하므로 원본 해상도 유지
방식보다 output 용량과 생성 시간이 크게 줄어든다.

## 6. 현재 진행 상태

2026-07-24에 `google_10`, `oppo_10`, `h8_10`, `q8_10`을 생성하고 image와 LabelMe JSON 10쌍씩의
형식, 개수, ROI geometry를 검증했다. 전체 40장 preview는 `data/augmented/preview_10.png`에 저장했다.
현재 사용자의 image 품질 및 annotation 승인을 기다리고 있으며, 승인 전에는 100장 생성을 실행하지 않는다.
