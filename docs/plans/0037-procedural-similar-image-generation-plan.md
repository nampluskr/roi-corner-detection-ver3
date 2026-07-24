# 0037 Procedural Similar Image Generation Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-24 |
| 적용 범위 | `data/make_similar_images.py`, `data/similar/` |
| 관련 문서 | [05-data-strategy.md](../architecture/05-data-strategy.md), [0036-measured-augmentation-plan.md](0036-measured-augmentation-plan.md) |

## 1. 목적과 배경

기존 `data/make_augmented_images.py`는 실제 TIFF를 입력으로 사용하므로 원본 data가 없는 환경에서 실행할 수
없다. google, h8, q8, oppo 측정 image의 공통 시각 특성을 절차적으로 렌더링하는 독립 generator를 추가해,
원본 파일 없이도 재현 가능한 similar dataset을 생성한다.

## 2. 범위

포함하는 작업은 다음과 같다.

- 단일 generator `data/make_similar_images.py`를 추가한다.
- google, h8, q8, oppo별 panel 크기, aspect ratio, corner radius, fringe frequency, fringe deformation,
  fixture와 jig 배치를 profile로 정의한다.
- 어두운 perforated stage, panel shadow, 수평 fringe, 국부 wave distortion, 둥근 panel corner, panel 외곽의
  clamp와 fringe tab을 절차적으로 렌더링한다.
- synthetic generator와 같이 outer rounded mask와 inset active rounded mask 사이를 어두운 bezel로 채워
  라운딩을 따라가는 검은 panel 테두리를 렌더링한다.
- generated TIFF에는 ROI quadrilateral 또는 정답 외곽선을 그리지 않고 rounded panel mask 경계만 표시한다.
- 실제 image의 문자열, sticker, camera 또는 고유 부품 형상은 재현하지 않는다.
- 기본 출력은 1920 x 1080 grayscale TIFF와 4점 `roi` polygon을 가진 LabelMe JSON으로 한다.
- `--count 10` 실행 시 `data/similar/google_10`, `h8_10`, `q8_10`, `oppo_10`을 생성한다.
- seed, count, width, height, category를 CLI option으로 제공하고 기존 output을 덮어쓰지 않는다.

제외하는 작업은 다음과 같다.

- `E:\fringe_data\training_all` 또는 다른 원본 image를 읽지 않는다.
- 기존 `data/augmented/`와 `data/make_augmented_images.py`를 변경하지 않는다.
- 문자열, sticker, camera와 source별 고유 물체를 생성하지 않는다.
- 검토용 10장 승인 전에는 category별 100장을 생성하지 않는다.

## 3. 완료 기준

- 원본 data 경로가 없는 환경에서도 `data/make_similar_images.py --count 10`이 실행된다.
- 네 category folder에 각각 TIFF와 LabelMe JSON 10쌍이 생성된다.
- 모든 image는 1920 x 1080 grayscale이고, JSON의 `roi`는 image 내부의 convex 4점 polygon이다.
- 동일한 seed와 인자로 실행하면 동일한 결과를 생성한다.
- 정답 polygon overlay가 없는 전체 preview에서 category별 panel 비율, corner rounding, fringe와 jig 차이가
  확인된다.

## 4. 검증

WinPython interpreter로 다음 명령을 실행한다.

```powershell
& C:\winpython\WPy64-31180\python-3.11.8.amd64\python.exe data\make_similar_images.py --count 10
```

실행 후 네 folder의 TIFF와 JSON 개수, image mode와 크기, LabelMe schema, ROI geometry를 검증한다.
전체 40장을 정답 polygon overlay 없이 한 화면에서 확인할 수 있는 `data/similar/preview_10.png`를 생성한다.

## 5. 완료 결과

2026-07-24에 source-independent generator를 구현하고 google, h8, q8, oppo별 검토용 image와 LabelMe JSON
10쌍을 생성했다. 네 category의 file set, image 형식, ROI geometry, seed 재현성과 file 고유성을 검증했고,
전체 결과를 정답 polygon overlay가 없는 `data/similar/preview_10.png`에서 확인했다.
