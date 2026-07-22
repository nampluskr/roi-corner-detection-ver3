# Synthetic Generation Guide

이 문서는 synthetic 단계 data를 만드는 방법을 설명한다. OLED 패널의 fringe image와 그 안의 ROI corner
네 점을 합성하고, LabelMe polygon JSON을 함께 생성하며, 그 결과를 학습용 gt_corners.csv로 변환하는
과정을 다룬다. synthetic 단계가 왜 필요한지와 합성 자동 레이블의 수학적 원리는 [Data Strategy](../architecture/05-data-strategy.md)에서
다루며, 이 문서는 그 전략을 실제로 실행하는 절차와 파라미터를 다룬다.

합성 data는 실측 data가 소량인 상황에서 model을 검사 domain의 fringe texture에 적응시키는 용도로
사용한다. 생성기는 정규화 사각형 위에 fringe pattern을 그린 뒤 homography로 원근 변환을 적용하므로,
corner label을 변환 parameter에서 오차 없이 자동으로 얻는다.

## 1. 산출물과 실행 스크립트

합성 생성은 두 스크립트로 나뉜다. 첫째는 image와 LabelMe JSON을 만드는 생성기이고, 둘째는 그 JSON을
gt_corners.csv로 변환하는 parser CLI다.

| 스크립트 | 역할 | 산출물 |
| --- | --- | --- |
| `data/make_synthetic_images.py` | fringe image와 LabelMe polygon JSON 생성 | `data/synthetic/synthetic_0X/*.png`, `*.json` |
| `data/make_gt_corners.py` | LabelMe JSON을 gt_corners.csv로 변환 | `data/synthetic/gt_corners.csv` |

생성기는 세 preview geometry profile을 한 번에 실행하여 각 결과를 다음 폴더에 저장한다.

| geometry profile | 출력 폴더 | 특성 |
| --- | --- | --- |
| `preview_01` | `data/synthetic/synthetic_01` | near-top-view, 큰 ROI, 면적 50% 이상 |
| `preview_02` | `data/synthetic/synthetic_02` | tilt 촬영, trapezoid ROI, 축소 패널과 좌우 shift |
| `preview_03` | `data/synthetic/synthetic_03` | preview_02 geometry에 multi-scale phase, 낮은 경계 대비, 다중 holder 추가 |

image와 JSON은 같은 폴더에 저장하며 항상 같은 stem을 사용한다. 이 구조에서는 LabelMe JSON의
`imagePath`를 image 파일명만으로 기록할 수 있다.

## 2. 참고 근거

fringe morphology와 장비 배치는 phase measuring deflectometry(PMD) 참고 자료에서 관찰한 특징을 따른다.
물체 표면에 명암 대비가 큰 단색 fringe가 나타나고, 반사면 형상과 촬영 시점에 따라 fringe가 완전히 곧지
않고 완만하게 휜다. 물체 내부와 가장자리에서 밝기와 대비가 균일하지 않으며, 프레임 바깥 배경과 반사
물체 사이의 경계가 뚜렷하다.

참고 자료의 어두운 배경은 fringe와 반사 물체의 외관을 설계하기 위한 사례이며, 배경 밝기를 검은색으로
제한하는 조건은 아니다. 실제 검사 환경에서는 주변 조명의 세기와 방향, 카메라 노출 때문에 배경이
검은색부터 밝은 회색까지 달라진다. 따라서 합성 data는 배경 평균 밝기와 공간적 조명 분포를 독립 변수로
사용하고, 밝은 배경에서 OLED 외곽 구분이 어려운 사례도 포함한다. 물체 형상은 참고 자료를 그대로
복사하지 않고 실제 목표인 rounded OLED rectangle 또는 rounded OLED square로 제한한다. scratch, pit,
파손 같은 표면 결함은 넣지 않는다. 카메라 hole은 제품 구조이므로 결함과 별개로 모든 image에 포함한다.

## 3. 변형 변수 계열

한 표본은 여러 변형 변수를 조합해 만든다. 각 변수는 독립적으로 표본 분포를 넓히는 축이며, 사용자는
필요에 따라 변수 계열별 변형 개수를 차별 적용한다. measured 분포에서 자주 나타나는 변형은 개수를
늘리고 거의 나타나지 않는 변형은 줄여 합성 표본 분포를 실측 분포에 가깝게 맞춘다.

변형 변수 계열과 범위는 다음과 같다. 범위 값은 baseline이며 생성 script에서 조정할 수 있다.

| 변수 계열 | 주요 옵션과 범위 |
| --- | --- |
| 위치와 자세 | rotation `$-10$`부터 `$+10$`도, 수평 이동 `$\pm 8\%$`, 수직 이동 `$\pm 2.5\%$`, top-bottom 비율 0.84부터 0.98의 trapezoid, panel 크기 preview_01의 약 80% |
| corner 라운딩 | 짧은 변 대비 반지름 3%부터 8%, 직사각형과 정사각형 panel |
| 외부 지그 | 배치 left-right, top-bottom, four sides, 변당 개수 1부터 3, 길이 6%부터 20%, 깊이 3%부터 12%, 밝기 0.06부터 0.50 |
| 카메라 hole | 위치 top-center, upper-left, 지름 3%부터 6%, 가시성 visible, partial, hidden |
| 배경 밝기 | 평균 0.03부터 0.90의 dark, medium, bright bin, 조명 gradient, vignetting 0%부터 25%, noise 1부터 8 gray |
| fringe 왜곡 | 주파수 8부터 36 cycle, 위상 `$\{0, \pi/2, \pi, 3\pi/2\}$`, 방향 vertical과 horizontal, global bow deformation |

각 변수 계열의 실제 합성 예시는 slide 자산에 있으며 파일 목록은 [Data Strategy](../architecture/05-data-strategy.md)의
합성 변형 변수 절에 정리되어 있다.

## 4. Fringe 모델

기본 fringe는 다음 sinusoidal 모델을 사용한다.

```text
I(u, v) = A + B * cos(2 * pi * f * axis(u, v) + phi + delta(u, v))
```

각 파라미터의 범위는 다음과 같다.

| 파라미터 | 범위 또는 값 |
| --- | --- |
| bias `A` | local stage mean을 기준으로 0.20부터 0.78 |
| amplitude `B` | soft 0.10부터 0.18, medium 0.18부터 0.28, clear 0.28부터 0.36 |
| frequency `f` | canonical OLED당 8부터 36 cycle, hidden hole은 8부터 12 |
| phase `phi` | `0`, `pi/2`, `pi`, `3*pi/2` 중 하나 |
| deformation | mild, moderate, severe |
| phase field | global bow, smooth 2D random field, 2부터 4개 RBF bump, spatial chirp |
| panel blur sigma | canonical 영상 기준 0.5부터 1.6 pixel |
| edge feather sigma | warped 영상 기준 1.5부터 4.0 pixel |

`A`와 `B`는 local stage 밝기와 clipping 조건에 맞게 함께 제한한다. fringe 변형은 연속 phase field로
생성하여 crack이나 phase discontinuity로 보이지 않게 한다. RGB 세 채널에는 같은 fringe를 넣되 채널별
gain을 0.97부터 1.03 범위에서 미세하게 달리하여 카메라 color response만 약하게 모사한다.

## 5. 카메라 hole 가시성 제어

카메라 hole은 OLED canonical 좌표계에서 검은 원으로 생성하고, fringe와 optical effect를 적용한 뒤
homography를 적용하기 전에 원 내부 픽셀을 검은색으로 교체한다. 따라서 perspective가 적용되면 원은 카메라
영상에서 타원에 가까워질 수 있다. 가시성은 우연히 정해지게 두지 않고 fringe의 주파수와 위상을
재샘플링하여 목표 상태에 맞춘다.

| 가시성 | 목표 상태 | 제어 방법 |
| --- | --- | --- |
| visible | 밝은 fringe 중심에 놓여 검은 hole이 선명함 | hole 중심을 fringe 최대 밝기 근처에 오도록 `phi`와 필요 시 `f`를 재샘플링 |
| partial | 명암 경계에 걸쳐 hole 일부만 선명함 | hole 중심을 밝고 어두운 띠의 zero-crossing 근처에 배치 |
| hidden | 검은 fringe 중심에 놓여 주변과 구분이 어려움 | hole 중심을 fringe 최소 밝기에 두고 hole 지름을 dark band 폭 이하로 제한 |

hidden은 hole을 삭제하는 경우가 아니다. 실제 검은 원은 항상 존재하지만 검은 fringe와의 낮은 대비 때문에
영상에서 거의 보이지 않게 만든다. hole은 image의 어려운 외관 조건일 뿐 검출 대상이 아니므로 LabelMe
shape로 추가하지 않는다.

## 6. Corner 레이블 정의

OLED의 실제 픽셀 외곽은 rounded rectangle이므로 모서리에 하나의 명확한 sharp pixel이 없다. model의 출력
계약은 네 점이므로 다음 기준으로 corner를 정의한다.

1. canonical OLED의 위, 오른쪽, 아래, 왼쪽 직선 변을 정의한다.
2. 인접한 직선 변을 둥근 구간 너머로 연장한다.
3. 연장한 직선의 네 교점을 TL, TR, BR, BL 가상 corner로 정의한다.
4. OLED 영상과 동일한 homography로 이 네 점을 destination quad로 변환한다.
5. 변환된 네 점을 LabelMe polygon에 픽셀 좌표로 기록한다.

둥근 원호의 중간점, 원호가 직선에 접하는 점, 카메라 hole의 중심은 corner 레이블로 사용하지 않는다.
rounded mask 때문에 가상 corner 픽셀 자체는 배경에 놓일 수 있지만, 이 점은 패널의 기준 사각형과 원근
복원에 필요한 일관된 기하 좌표다.

## 7. 생성 파이프라인

한 표본은 다음 순서로 생성한다.

1. 사용자 seed와 표본 번호로 local RNG를 초기화한다.
2. 표본 번호에 대응하는 OLED 형상, fringe 방향, 배경, hole 위치와 가시성을 선택한다.
3. 평탄한 stage 표면을 생성하고 지정된 조명 밝기와 공간적 illumination 조건을 적용한다.
4. 투영 전 canonical OLED reference rectangle을 픽셀 좌표로 정의하고 rounded mask를 만든다.
5. 8부터 36 cycle의 base phase를 생성하고 global bow, 2D random field, RBF bump, spatial chirp를 합성한다.
6. deformation과 contrast tier, 위치별 illumination gradient, modulation, blur, noise를 적용한다.
7. 지정된 위치와 가시성 조건에 맞게 camera hole을 합성한다.
8. reference rectangle을 축소하고 윗변을 줄여 약한 trapezoid를 만든 뒤 좌우 shift와 rotation으로 destination quad를 만든다.
9. OLED와 mask에 동일 homography를 적용하고 경계를 feather 합성한다.
10. 지정된 변과 개수에 맞춰 holder를 OLED 가장자리에 접촉하도록 합성한다.
11. destination quad를 LabelMe `roi` polygon의 가상 네 corner로 사용한다.
12. PNG를 먼저 저장하고 동일 stem의 JSON을 저장한 뒤 기하, phase, 경계 대비, holder 조건을 사후 검증한다.

중간 단계에서 기하 조건 또는 hole 가시성 조건을 만족하지 못하면 해당 표본의 파라미터를 재샘플링한다.
한 표본당 여러 번 시도하고 모두 실패하면 불완전한 파일을 남기지 않고 명시적 오류로 종료한다.

## 8. LabelMe JSON 규격

생성기는 LabelMe 라이브러리를 실행 의존성으로 두지 않고 공식 JSON 구조를 직접 기록한다. 저장 규칙은
다음과 같다.

- JSON 파일명은 대응 image와 같은 stem을 사용한다.
- `imagePath`는 디렉터리가 없는 PNG 파일명만 저장한다.
- `imageData`는 `null`로 저장한다.
- `imageWidth`와 `imageHeight`는 실제 PNG 크기와 일치한다.
- `shapes`에는 `roi` polygon 하나만 둔다.
- `points`는 TL, TR, BR, BL 순서의 float pixel coordinate 네 개다.

JSON 예시는 다음과 같다.

```json
{
  "version": "6.2.0",
  "flags": {},
  "shapes": [
    {
      "label": "roi",
      "points": [
        [182.345678, 53.123456],
        [1736.234567, 69.456789],
        [1761.456789, 1021.234567],
        [159.123456, 1008.345678]
      ],
      "group_id": null,
      "description": "",
      "shape_type": "polygon",
      "flags": {},
      "mask": null
    }
  ],
  "imagePath": "synthetic_0001.png",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

합성 파라미터, fringe 방향, OLED 형상, hole 가시성은 JSON의 custom field나 `flags`에 넣지 않는다. 향후
실측 data에서 LabelMe가 생성하는 JSON과 레이블 구조를 같게 유지하기 위함이다.

## 9. 실행 방법

image와 JSON 생성은 다음 command로 실행한다. 세 preview 폴더를 한 번에 만든다.

```bash
python data/make_synthetic_images.py \
    --count 20 \
    --width 1920 \
    --height 1080 \
    --seed 42 \
    --background-dir /mnt/d/datasets/dtd/images
```

인수 규칙은 다음과 같다.

| 인수 | 기본값 | 규칙 |
| --- | --- | --- |
| `--count` | `20` | 1부터 9999 |
| `--width` | `1920` | 320 이상 |
| `--height` | `1080` | 320 이상 |
| `--seed` | `42` | local RNG 초기값 |
| `--background-dir` | 생략 가능 | DTD 등 배경 image 루트, 생략하면 절차식 texture 사용 |

출력 폴더가 비어 있지 않으면 덮어쓰지 않고 오류로 종료한다. 재생성하려면 대상 `synthetic_0X` 폴더를
먼저 비운다.

생성한 LabelMe JSON을 gt_corners.csv로 변환하려면 다음 command를 실행한다. `--data_dir`은 image와 JSON이
있는 폴더이며, 하위 폴더까지 재귀로 탐색한다.

```bash
python data/make_gt_corners.py \
    --dataset labelme \
    --data_dir data/synthetic \
    --output_path data/synthetic/gt_corners.csv
```

`data/synthetic`을 지정하면 세 하위 폴더의 JSON을 모두 수집한다. 변환 결과 CSV의 schema와 corner 순서
검증은 [Dataset Format Guide](01-dataset-format.md)를 따른다.

## 10. Code mapping

합성 생성과 변환에 관여하는 파일은 다음과 같다.

| 관심사 | 파일 |
| --- | --- |
| fringe image와 LabelMe JSON 생성 엔진 | `data/make_synthetic_images.py` |
| LabelMe JSON을 gt_corners.csv로 변환 | `src/data/labelme.py` |
| 변환 CLI 진입점 | `data/make_gt_corners.py` |
| corner 순서 정규화와 유효성 검사 | `src/utils/geometry.py` |
| 변환 후 CSV load와 split | `src/data/dataset.py` |

## 11. 핵심 요약

synthetic 단계는 `data/make_synthetic_images.py`로 세 preview geometry의 fringe image와 LabelMe polygon을
생성하고, `data/make_gt_corners.py`의 labelme parser로 그 JSON을 gt_corners.csv로 변환한다. fringe는
sinusoidal 모델에 phase field 변형을 더해 만들며, corner label은 rounded OLED의 직선 변을 연장한 가상
교점을 homography로 변환해 오차 없이 얻는다. 카메라 hole 가시성은 fringe 위상을 재샘플링하여 visible,
partial, hidden 세 단계로 제어한다. image와 JSON은 같은 stem으로 같은 폴더에 저장하고, LabelMe JSON은
실측 data와 동일한 `roi` polygon 구조를 유지한다.
