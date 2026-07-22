# Synthetic OLED Fringe 이미지 생성 계획

> 주의: `data/`는 프로젝트의 `.gitignore` 대상이다. 따라서 이 문서는 현재 위치에서는
> Git으로 추적되지 않는다. 문서를 다른 환경과 공유하거나 버전 관리해야 할 경우에는
> 별도의 추적 경로로 복사하는 절차가 필요하다.

## 1. 목적과 범위

이 문서는 OLED 디스플레이 패널의 네 코너를 검출하기 위한 합성 PMD(Phase Measuring
Deflectometry) fringe 이미지와 LabelMe 레이블 생성 방법을 정의한다. 합성 데이터는
실측 데이터가 적은 상황에서 모델을 fringe 도메인에 적응시키는 용도로 사용한다.

현재 작업 범위는 다음과 같다.

- `data/synthetic/preview_03/`에 검토용 이미지 20장을 생성한다.
- 각 이미지와 동일한 stem의 LabelMe JSON 파일을 생성한다.
- 이미지는 1920x1080 RGB PNG로 저장하되 시각적 내용은 흑백으로 구성한다.
- LabelMe에는 `roi` polygon 하나만 저장한다.
- `gt_corners.csv`, 학습 설정, 모델, 기존 데이터 파이프라인은 변경하지 않는다.
- preview 검토가 끝나기 전에는 대규모 데이터를 생성하지 않는다.

`preview_01`은 top-view에 가까운 큰 ROI를 사용한 초기 비교용 버전으로 보존한다.
`preview_02`는 tilt 촬영, trapezoid ROI, 축소된 패널 크기와 좌우 shift를 반영한
비교용 버전이다. `preview_03`은 `preview_02` geometry를 유지하고 실측 참고 fringe,
낮은 ROI 경계 대비, 위치별 조명 변화와 다중 holder를 추가한 현재 검토 대상이다.

preview 이후에는 같은 생성기를 사용하되 결과를 `data/synthetic/{condition}/`에 조건별로
분리한다. 최종 조건명과 조건별 수량은 preview 검토 후 확정한다.

## 2. 참고 논문과 반영 원칙

### 2.1 참고 논문

1. G. Qiao et al., "A single-shot phase retrieval method for phase measuring
   deflectometry based on deep learning," Optics Communications, 476, 126303, 2020.
   - DOI: https://doi.org/10.1016/j.optcom.2020.126303
2. L. Fan et al., "Deep learning-based Phase Measuring Deflectometry for single-shot
   3D shape measurement and defect detection of specular objects," Optics Express,
   30(15), 26504-26518, 2022.
   - DOI: https://doi.org/10.1364/OE.464452

### 2.2 시각적 특징

첫 번째 논문의 Fig. 2와 Fig. 6에서 다음 특징을 참고한다. 논문 예시의 어두운 배경은
fringe와 반사 물체의 외관을 설계하기 위한 참고 사례이며, 실제 데이터의 배경 밝기를
검은색으로 제한하는 조건은 아니다.

- 배경은 검은색 또는 매우 어두운 회색이다.
- 물체 표면에는 명암 대비가 큰 단색 fringe가 나타난다.
- 반사면의 형상과 촬영 시점에 따라 fringe가 완전히 곧지 않고 완만하게 휜다.
- 물체 내부와 가장자리에서 밝기와 대비가 균일하지 않다.
- 프레임 바깥 배경과 fringe가 표시되는 반사 물체 사이의 경계가 뚜렷하다.

실제 검사 환경에서는 주변 조명의 세기와 방향, 카메라 노출 및 반사체 때문에 배경이
검은색부터 밝은 회색까지 달라질 수 있다. 따라서 합성 데이터의 배경 평균 밝기와 공간적
조명 분포를 독립 변수로 사용한다. OLED와 배경 사이의 대비도 항상 높게 고정하지 않고,
밝은 배경에서 OLED 외곽 구분이 어려운 사례를 preview에 포함한다.

두 번째 논문의 mobile phone glass panel 데이터에서 다음 특징을 참고한다.

- iPhone 및 Samsung mobile phone glass panel처럼 OLED와 가까운 평판형 외형을 사용한다.
- 단일 흑백 distorted fringe를 입력 영상의 기본 형태로 삼는다.
- 실제 카메라 영상처럼 약한 blur, sensor noise, gamma 및 국부적인 대비 저하가 존재한다.
- 서로 다른 패널은 표면의 미세한 기울기와 반사 특성 때문에 fringe 변형 정도가 다르다.

논문의 물체 형상을 그대로 복사하지 않는다. 원형 거울, 자유곡면, 큰 구멍 등은 제외하고
실제 목표인 rounded OLED rectangle 또는 rounded OLED square로 변형한다. 논문에서 다룬
scratch, pit, 파손 및 표면 결함도 preview에는 넣지 않는다. 카메라 홀은 제품 구조이므로
결함과 별개로 모든 이미지에 포함한다.

### 2.3 실측 장비 환경

`data/samples/pmd_system_stage.jpg`를 실제 사용자의 측정 환경과 가장 가까운 장비
참고 이미지로 사용한다. 합성 영상은 다음 배치를 기본으로 한다.

- OLED 패널은 평면 stage 위에 놓인다.
- 영상에서 stage 표면과 주변 장비 구조가 보일 수 있다.
- 참고 사진의 stage 위에 있는 검은색 종이와 종이 테두리는 합성하지 않는다.
- OLED는 서로 마주 보는 2개의 jig 또는 rubber block으로 고정한다.
- 고정 방향은 left-right와 top-bottom을 모두 생성 변수로 사용한다.

고정 block이 bezel만 접촉하는지 active fringe 영역을 일부 가리는지는 아직
확정되지 않았다. 이 항목을 확정하기 전까지 preview에서는 block이 OLED 외곽과
접촉하되 active fringe를 가리지 않게 생성하며 README의 F8은 변경하지 않는다.

### 2.4 실측 fringe 참고 이미지 분석

`data/samples/pmd_measurement_01.png`-`pmd_measurement_05.png`는 원본 OLED 실측 프레임만으로
구성된 데이터가 아니라 schematic, 실험 장비 화면과 논문 figure가 섞여 있다.
따라서 배경, 노이즈와 해상도를 그대로 복제하지 않고 fringe morphology만 참고한다.

| 파일 | 관찰 특징 | `preview_03` 반영 방법 |
|---|---|---|
| `pmd_measurement_01.png` | 이상적 직선과 촬영된 S-curve, global bow, 간격 압축과 팽창 비교 | global bow와 spatial chirp |
| `pmd_measurement_02.png` | 장비 화면의 phase, 굵기, 주기와 대비 변화 | coarse, medium, dense frequency와 contrast tier |
| `pmd_measurement_03.png` | 다중 스케일 굴곡, 국부 간격 변화, grain과 대비 저하 | 2D smooth phase field와 modulation field, 노이즈는 약화해 적용 |
| `pmd_measurement_04.png` | 국부 돌기 주변의 큰 수렴, 발산과 U-shape | 낮은 비율의 severe RBF deformation, phase 단절은 제외 |
| `pmd_measurement_05.png` | 수평과 수직 fringe의 완만한 굴곡, 방향 변화와 불균일 간격 | mild/moderate 표본의 기본 deformation |

중앙 crop의 상대 주기는 `01` 10-12 cycles, `03` 약 17 cycles, `04` 약 9 cycles,
`05` 약 19와 24 cycles로 관찰되었다. 해상도와 crop이 다르므로 이 값은 절대
픽셀 주기가 아니라 다양성 설계 근거로만 사용한다.

## 3. 출력 구조와 파일명

### 3.1 폴더 구조

```text
data/
└── synthetic/
    ├── synthetic-image-generation-plan.md
    └── preview_03/
        ├── synthetic_0001.json
        ├── synthetic_0001.png
        ├── synthetic_0002.json
        ├── synthetic_0002.png
        └── ...
```

이미지와 JSON은 같은 폴더에 저장하며 반드시 같은 stem을 사용한다. 이 구조에서는
LabelMe JSON의 `imagePath`를 이미지 파일명만으로 기록할 수 있다.

### 3.2 번호 규칙

- 파일명 형식은 `synthetic_NNNN`이다.
- 번호는 각 condition 폴더에서 `0001`부터 다시 시작한다.
- 번호는 항상 4자리이며 허용 범위는 `0001`부터 `9999`이다.
- condition 하나에 9,999장을 초과하여 생성하지 않는다.
- 같은 condition 폴더가 비어 있지 않으면 생성하지 않고 오류로 종료한다.
- 기존 PNG 또는 JSON을 자동으로 덮어쓰거나 이어 쓰지 않는다.

## 4. Preview_03 20장 구성

### 4.1 OLED 형상과 fringe 방향

| 번호 | OLED 형상 | 보이는 fringe 방향 | 배경 순서 |
|---|---|---|---|
| `0001`-`0005` | rectangle | horizontal | 5종 순서 적용 |
| `0006`-`0010` | rectangle | vertical | 5종 순서 적용 |
| `0011`-`0015` | square | horizontal | 5종 순서 적용 |
| `0016`-`0020` | square | vertical | 5종 순서 적용 |

`horizontal`은 화면에 보이는 띠가 가로 방향이라는 뜻이며 밝기는 y축을 따라 변한다.
`vertical`은 화면에 보이는 띠가 세로 방향이라는 뜻이며 밝기는 x축을 따라 변한다.
수식의 변화 축이 아니라 사용자가 영상에서 보는 줄 방향을 이름으로 사용한다.

### 4.2 패널 크기와 tilt 촬영 자세

`preview_03`은 `preview_02`의 geometry를 그대로 사용한다. 패널 가로와 세로 길이는
`preview_01`의 약 80% 수준이다.
선형 크기를 80%로 조정하므로 reference quad 면적은 기존의 약 64%가 된다.

- 카메라는 stage에 수직인 top-view가 아니라 기울어진 시점에서 OLED를 촬영한다.
- 원근에 의해 OLED의 윗변이 아랫변보다 짧게 투영된 약한 trapezoid를 사용한다.
- projected top/bottom width ratio는 0.84-0.98이다.
- 패널 중심은 영상 중앙을 기준으로 하되 이미지 너비의 최대 +-8%까지 좌우로 shift한다.
- 수직 위치는 이미지 높이의 +-2.5% 범위에서만 변화시킨다.
- in-plane rotation은 -10 to +10 degrees 범위로 생성한다.
- 크기, trapezoid ratio, shift, rotation은 20장에 고르게 분산하여 모든 표본이
  유사해지지 않게 한다.

reference quad 면적은 rectangle에서 약 34-44%, square에서 약 31-36%를 목표로
한다. 이 범위는 README의 현재 F2 `이미지의 50% 이상`을 만족하지 않는다.
`preview_03` 검토 후 실측 ROI 크기와 유사하다고 승인되면 F2와 학습 데이터
검증 기준을 함께 갱신한다.

### 4.3 배경 순서

각 5장 그룹에는 아래 배경을 같은 순서로 한 번씩 사용한다.

| 그룹 내 순서 | 배경 | 설명 |
|---:|---|---|
| 1 | dark stage | 평균 밝기 0.03-0.18의 어두운 무채색 stage 표면 |
| 2 | medium stage | 평균 밝기 0.25-0.50의 중간 밝기 stage 표면 |
| 3 | bright illuminated stage | 평균 밝기 0.60-0.90의 밝은 조명이 비춘 stage 표면 |
| 4 | stage illumination gradient | 어두운 영역과 밝은 영역이 함께 있는 방향성 stage gradient |
| 5 | textured stage | DTD crop을 저대비 무채색 stage 미세 texture로 사용하고 노출을 보정한 배경 |

DTD 배경은 `/mnt/d/datasets/dtd/images` 아래의 `.jpg`, `.jpeg`, `.png` 파일을 재귀적으로
탐색한다. 파일 목록을 정렬한 뒤 seeded RNG로 선택하고, 1920x1080을 채우도록 aspect-ratio
preserving resize와 random crop을 적용한다. DTD 디렉터리를 지정하지 않으면 다섯 번째
배경도 절차식 texture로 대체한다. DTD는 의류, 자연물, 종이처럼 stage와 다른
구조를 그대로 표현하는 배경으로 사용하지 않는다. 저주파 성분과 채도를 억제해
평탄한 산업용 stage의 미세 표면 요철로만 사용한다. 목표 평균 밝기는 0.12-0.85에서
샘플링하며 목표값에 맞게 exposure를 보정한다.

배경 밝기는 단순 RGB offset이 아니라 exposure gain, gamma, low-frequency illumination을
조합해 만든다. 최종 배경은 RGB 각 채널을 [0, 1]로 제한하고, 지나친 clipping으로 넓은
영역이 완전한 흰색이 되지 않도록 99 percentile이 0.98 이하가 되게 한다.

### 4.4 Stage와 고정 block 구성

- left-right only: `0001`, `0002`, `0006`, `0010`, `0014`, `0015`, `0019`
- top-bottom only: `0003`, `0004`, `0008`, `0011`, `0013`, `0017`, `0018`
- four sides: `0005`, `0007`, `0009`, `0012`, `0016`, `0020`
- 접촉하는 맞은편 두 변은 같은 개수의 holder를 사용하며 한 변당 1-3개를 허용함
- four sides에서 left-right count와 top-bottom count는 다를 수 있음
- holder length는 접촉 변의 6-20%, 외부 돌출 깊이는 OLED 짧은 변의 3-12%
- 형상은 rectangle에서 outer taper ratio 0.60의 trapezoid까지 변화
- normalized brightness는 0.06-0.50이며 holder 내부에 방향성 gradient를 적용
- 다중 holder는 변의 15-85% 구간에 균등 배치한 뒤 작은 position jitter를 적용하고 서로 겹치지 않음
- 한 이미지의 holder는 공통 재질과 기본 형상을 공유하되 개별 길이와 밝기에 jitter를 적용
- edge highlight, 접촉 그림자와 선택적 screw를 표현
- holder는 OLED 외곽과 접촉하지만 active fringe를 가리지 않음
- 검은색 종이, 종이 테두리, 독립된 종이 받침 mask는 생성하지 않음

### 4.5 카메라 홀 위치

- 홀수 번호: `top-center`
- 짝수 번호: `upper-left`
- 결과적으로 각 위치는 10장씩 포함된다.

`top-center`는 OLED 상단 중앙을 기준으로 좌우 3% 범위에서 이동시킨다. `upper-left`는
OLED 좌측에서 폭의 12-22%, 상단에서 높이의 5-10% 범위에 배치한다. 카메라 홀 전체가
rounded OLED의 유효 화면 안에 남아 있어야 한다.

### 4.6 카메라 홀 가시성

| 가시성 | 번호 | 목표 상태 |
|---|---|---|
| visible | `0001`, `0004`, `0007`, `0010`, `0013`, `0016`, `0019` | 밝은 fringe 중심에 놓여 검은 홀이 선명함 |
| partial | `0002`, `0005`, `0008`, `0011`, `0014`, `0017`, `0020` | 명암 경계에 걸쳐 홀 일부만 선명함 |
| hidden | `0003`, `0006`, `0009`, `0012`, `0015`, `0018` | 검은 fringe 중심에 놓여 주변과 구분이 어려움 |

가시성은 결과적으로 우연히 정해지게 두지 않고 fringe의 주파수와 위상을 재샘플링하여
맞춘다. 홀 중심 주변의 정규화된 fringe 밝기는 visible에서 0.75 이상, partial에서
0.35-0.65, hidden에서 0.15 이하를 목표로 한다. hidden에서는 홀 반경을 local dark-band
폭의 35% 이하로 제한해 홀 가장자리가 인접한 밝은 띠와 닿지 않도록 한다.

## 5. 이미지 생성 파이프라인

한 표본은 아래 순서로 생성한다.

1. 사용자 seed와 표본 번호로 local RNG를 초기화한다.
2. 표본 번호에 대응하는 OLED 형상, fringe 방향, 배경, 홀 위치와 가시성을 선택한다.
3. 평탄한 산업용 stage 표면을 생성하고 지정된 조명 밝기와 공간적 illumination
   조건을 적용한다. 필요하면 저대비 DTD 미세 texture를 표면에만 합성한다.
4. 투영 전 canonical OLED reference rectangle을 픽셀 좌표로 정의한다.
5. reference rectangle과 같은 크기의 rounded rectangle mask를 생성한다.
6. 8-36 cycles의 horizontal 또는 vertical fringe base phase를 생성한다.
7. global bow, smooth 2D random field, 2-4개 RBF bump와 spatial chirp를 phase에 합성한다.
8. mild, moderate, severe 변형과 soft, medium, clear contrast tier를 적용한다.
9. 위치별 illumination gradient, modulation, blur와 noise를 적용한다.
10. 지정된 위치와 가시성 조건에 맞게 camera hole을 합성한다.
11. reference rectangle의 선형 크기를 `preview_01`의 약 80%로 축소한다.
12. 원근 시점에 맞게 윗변을 축소하여 약한 trapezoid를 만든다.
13. 패널을 영상 중앙에서 좌우로 shift하고 -10 to +10 degrees rotation을 적용해
    destination quad를 만든다.
14. OLED와 mask에 동일 homography를 적용한다.
15. premultiplied panel과 mask를 1.5-4.0 px로 함께 blur하여 경계를 normalized feather 합성한다.
16. 지정된 변과 개수에 맞춰 holder를 OLED 가장자리에 접촉하도록 합성한다.
17. destination quad를 LabelMe `roi` polygon의 가상 네 코너로 사용한다.
18. PNG를 먼저 저장하고 동일 stem의 JSON을 저장한다.
19. PNG/JSON, 기하, phase, 경계 대비, stage와 holder 조건을 사후 검증한다.

중간 단계에서 기하 조건 또는 홀 가시성 조건을 만족하지 못하면 해당 표본의 파라미터를
재샘플링한다. 한 표본당 최대 100회 시도하고, 모두 실패하면 불완전한 파일을 남기지 않고
명시적인 오류로 종료한다.

## 6. 파라미터 명세

### 6.1 공통 파라미터

| 항목 | preview 값 | 설명 |
|---|---|---|
| image size | 1920x1080 | 저장 원본 해상도 |
| image format | PNG RGB | 채널은 RGB, 내용은 grayscale appearance |
| count | 20 | 사용자 검토용 |
| seed | 42 | preview 재현성 기준 |
| panel linear size | `preview_01`의 약 80% | ROI 과대 문제 보정 |
| rotation | -10 to +10 degrees | 카메라 roll과 제품 회전 |
| horizontal shift | 이미지 너비의 최대 +-8% | 중앙 기준 좌우 이동 |
| vertical shift | 이미지 높이의 최대 +-2.5% | 중앙 근처 유지 |
| top/bottom width ratio | 0.84-0.98 | tilt 촬영에 의한 약한 trapezoid |
| corner jitter | OLED 크기의 축별 최대 0.4% | 투영 비대칭성 보조 |
| reference quad area | 전체 이미지의 30-46% | 가상 sharp-corner quad 기준 |
| corner order | TL -> TR -> BR -> BL | 시계 방향 |

### 6.2 OLED 형상

| 항목 | 범위 |
|---|---|
| rectangle aspect ratio `width / height` | 1.5-1.9 |
| square aspect ratio `width / height` | 0.95-1.05 |
| rounded corner radius | canonical OLED 짧은 변의 3-8% |
| camera-hole diameter | canonical OLED 짧은 변의 3-6% |

`preview_02`와 `preview_03`에서 rectangle reference quad는 이미지 면적의 약 34-44%, square는
약 31-36%가 되도록 샘플링한다. 둥근 모서리 때문에 실제 발광 mask 면적은
reference quad보다 조금 작다. 레이블과 면적 검증은 가상 sharp-corner reference quad를
기준으로 한다.

### 6.3 Stage와 고정 block

| 항목 | preview 값 |
|---|---|
| support surface | 평탄한 산업용 stage |
| black paper | 사용하지 않음 |
| fixture layout | left-right 7장, top-bottom 7장, four sides 6장 |
| fixture count | 접촉하는 한 변당 1-3개, 맞은편은 같은 개수 |
| fixture position | 변의 15-85% 구간 균등 배치와 +-3% jitter |
| block length | 접촉하는 변 길이의 6-20% |
| block depth | OLED 짧은 변의 3-12% |
| block taper ratio | 0.60-1.00 |
| block normalized brightness | 0.06-0.50 |
| block appearance | directional gradient, edge highlight, shadow, optional screw |
| active fringe occlusion | 0%, 사용자 확정 전 preview 기준 |

### 6.4 배경 밝기와 조명

| 항목 | 범위 또는 값 |
|---|---|
| normalized background mean | 0.03-0.90 |
| dark mean | 0.03-0.18 |
| medium mean | 0.25-0.50 |
| bright mean | 0.60-0.90 |
| exposure gain | 0.5-1.8 |
| gamma | 0.7-1.4 |
| gradient peak-to-peak | 0.10-0.55 |
| background vignetting | 0-25% |
| background noise sigma | 1-8 gray levels on [0, 255] |

배경의 목표 mean을 먼저 정하고 exposure gain과 gamma를 적용한 뒤 low-frequency
illumination을 더한다. gradient는 linear, radial 또는 두 성분의 합으로 만들며 광원의
방향과 중심을 RNG로 선택한다. 조명 변화는 OLED 내부 fringe의 밝기 변수와 별도로
샘플링하여, 밝은 배경과 어두운 OLED 또는 어두운 배경과 밝은 OLED가 모두 나타나게 한다.

preview에서는 각 OLED/fringe 조합마다 dark, medium, bright 표본을 최소 한 장씩 포함한다.
최종 대규모 데이터에서는 dark, medium, bright brightness bin을 가능한 한 균등하게
배분하고, gradient와 DTD가 각 bin에 고르게 포함되도록 한다.

### 6.5 Fringe와 광학 효과

기본 fringe는 다음 모델을 사용한다.

```text
I(u, v) = A + B * cos(2 * pi * f * axis(u, v) + phi + delta(u, v))
```

| 파라미터 | 범위 또는 값 |
|---|---|
| bias `A` | local stage mean을 기준으로 0.20-0.78 |
| amplitude `B` | soft 0.10-0.18, medium 0.18-0.28, clear 0.28-0.36 |
| frequency `f` | 8-36 cycles per canonical OLED, hidden hole은 8-12 |
| phase `phi` | `0`, `pi/2`, `pi`, `3*pi/2` 중 하나 |
| deformation | mild 8장, moderate 8장, severe 4장 |
| phase field | global bow, smooth 2D random field, 2-4 RBF bumps, spatial chirp |
| illumination gradient | full 16장 0.08-0.35, weak 4장 0.02-0.06 peak-to-peak |
| panel blur sigma | 0.5-1.6 pixels, canonical 영상 기준 |
| edge feather sigma | 1.5-4.0 pixels, warped 영상 기준 |
| Gaussian noise sigma | 1-8 gray levels on [0, 255] |

`A`와 `B`는 local stage 밝기와 clipping 조건에 맞게 함께 제한한다. fringe 변형은
연속적인 phase field로 생성하여 crack, pit이나 phase discontinuity로 보이지 않게 한다.
RGB 세 채널에는 같은 fringe를 넣되
채널별 gain을 0.97-1.03 범위에서 미세하게 달리하여 카메라 color response만 약하게
모사한다.

## 7. 카메라 홀 생성과 가시성 제어

카메라 홀은 OLED canonical 좌표계에서 완전한 검은 원으로 생성한다. fringe와 optical
effect를 적용한 뒤, homography를 적용하기 전에 원 내부 픽셀을 검은색으로 교체한다.
따라서 perspective가 적용되면 원은 카메라 영상에서 자연스럽게 타원에 가까워질 수 있다.

가시성별 처리 규칙은 다음과 같다.

- `visible`: 홀 중심이 fringe 최대 밝기 근처에 오도록 `phi`와 필요 시 `f`를 재샘플링한다.
- `partial`: 홀 중심을 밝고 어두운 띠의 zero-crossing 근처에 둔다.
- `hidden`: 홀 중심을 fringe 최소 밝기에 두고 홀 지름이 dark band를 넘지 않도록 제한한다.
- 모든 경우에 홀 전체가 rounded OLED mask 내부에 있어야 한다.
- 홀은 이미지의 어려운 외관 조건일 뿐 검출 대상이 아니므로 LabelMe shape를 추가하지 않는다.

hidden은 홀을 삭제하는 경우가 아니다. 실제 검은 원은 항상 존재하지만 검은 fringe와의
낮은 대비 때문에 영상에서 거의 보이지 않게 만든다. 검증 단계에서는 생성 당시 보관한
hole mask를 사용하여 존재 여부를 확인하고, 최종 산출물에는 이 mask를 저장하지 않는다.

## 8. Corner 레이블 정의

OLED의 실제 픽셀 외곽은 rounded rectangle이므로 모서리에 하나의 명확한 sharp pixel이
없다. 모델의 출력 계약은 네 점이므로 다음 기준을 사용한다.

1. canonical OLED의 위, 오른쪽, 아래, 왼쪽 직선 변을 정의한다.
2. 인접한 직선 변을 둥근 구간 너머로 연장한다.
3. 연장한 직선의 네 교점을 TL, TR, BR, BL 가상 코너로 정의한다.
4. OLED 영상과 동일한 homography로 이 네 점을 destination quad로 변환한다.
5. 변환된 네 점을 LabelMe polygon에 픽셀 좌표로 기록한다.

둥근 원호의 중간점, 원호가 직선에 접하는 점, 카메라 홀의 중심은 corner 레이블로 사용하지
않는다. rounded mask 때문에 가상 코너 픽셀 자체는 배경에 놓일 수 있지만, 이 점은 패널의
기준 사각형과 원근 복원에 필요한 일관된 기하 좌표이다.

## 9. LabelMe JSON 규격

### 9.1 저장 규칙

- JSON 파일명은 대응 이미지와 같은 stem을 사용한다.
- `imagePath`는 디렉터리가 없는 PNG 파일명만 저장한다.
- `imageData`는 LabelMe 6 기본 방식에 맞춰 `null`로 저장한다.
- `imageWidth`와 `imageHeight`는 실제 PNG 크기와 일치해야 한다.
- `shapes`에는 `roi` polygon 하나만 둔다.
- `points`는 TL, TR, BR, BL 순서의 float pixel coordinate 네 개이다.
- 좌표는 JSON에서 소수점 6자리까지 기록한다.

### 9.2 JSON 예시

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

합성 파라미터, fringe 방향, OLED 형상, 홀 가시성은 JSON의 custom field 또는 `flags`에
넣지 않는다. 향후 실측 데이터에서 LabelMe가 생성하는 JSON과 레이블 구조를 같게 유지하기
위함이다. preview 구성은 표본 번호 규칙과 생성 로그로 검증한다.

## 10. 독립 생성기 구현 계획

### 10.1 파일과 의존성

생성기는 preview 버전을 명확히 구분하기 위해 공통 엔진과 세 실행 파일로
구현한다.

```text
scripts/
├── generate_synthetic_labelme.py
├── generate_synthetic_preview_01.py
├── generate_synthetic_preview_02.py
└── generate_synthetic_preview_03.py
```

- `generate_synthetic_labelme.py`: 공통 stage, OLED, fringe, block, LabelMe 렌더링 엔진
- `generate_synthetic_preview_01.py`: 큰 ROI, near-top-view, 면적 50% 이상 profile
- `generate_synthetic_preview_02.py`: 약 80% 선형 크기, tilt, trapezoid, 좌우 shift profile
- `generate_synthetic_preview_03.py`: `preview_02` geometry, multi-scale phase,
  softened boundary, multi-holder profile

세 preview 실행 파일은 공통 엔진을 import하지만 profile과 기본 condition을
각각 `preview_01`, `preview_02`, `preview_03`으로 고정한다. 기존 프로젝트 소스나 설정 파일은
변경하지 않는다.

허용 의존성은 다음과 같다.

- Python standard library: `argparse`, `json`, `math`, `os`, `re`
- NumPy: 수식, 좌표, noise 및 mask 계산
- Pillow: 이미지 로딩, RGB 변환 및 PNG 저장
- OpenCV: homography, warp, blur 및 mask 처리

LabelMe 라이브러리는 실행 의존성으로 두지 않는다. 공식 JSON 구조를 직접 기록하여
LabelMe에서 열 수 있는 파일을 생성한다.

### 10.2 CLI

```text
python scripts/generate_synthetic_preview_01.py \
    --count 20 \
    --width 1920 \
    --height 1080 \
    --seed 42 \
    --background-dir /mnt/d/datasets/dtd/images
```

```text
python scripts/generate_synthetic_preview_02.py \
    --count 20 \
    --width 1920 \
    --height 1080 \
    --seed 42 \
    --background-dir /mnt/d/datasets/dtd/images
```

```text
python scripts/generate_synthetic_preview_03.py \
    --count 20 \
    --width 1920 \
    --height 1080 \
    --seed 42 \
    --background-dir /mnt/d/datasets/dtd/images
```

고급 용도에서는 공통 엔진을 `--geometry-profile preview_01`,
`--geometry-profile preview_02` 또는 `--geometry-profile preview_03`으로 직접 실행할 수 있다.
일반 preview 생성은
버전 혼동을 막기 위해 전용 실행 파일을 사용한다.

| 인수 | 기본값 | 규칙 |
|---|---|---|
| `--condition` | 실행 파일에 따라 `preview_01`, `preview_02`, `preview_03` | 영문, 숫자, `_`, `-`만 허용 |
| `--count` | `20` | 1-9999 |
| `--width` | `1920` | 양의 정수 |
| `--height` | `1080` | 양의 정수 |
| `--seed` | `42` | local RNG 초기값 |
| `--background-dir` | 생략 가능 | DTD 또는 다른 배경 이미지 루트 |
| `--geometry-profile` | 공통 엔진 직접 실행 시 `preview_02` | `preview_01`, `preview_02`, `preview_03` |

출력 루트는 프로젝트 기준 `data/synthetic`으로 고정하고 실제 출력은
`data/synthetic/{condition}`으로 계산한다. condition에 `/`, `..`, 공백 또는 기타 문자가
포함되면 경로를 만들지 않고 오류로 종료한다.
전용 실행 파일은 다른 geometry profile로 변경하는 CLI 인수를 노출하지 않는다.
기존 condition 폴더가 비어 있지 않으면 덮어쓰지 않고 오류로 종료한다.

`count=20`일 때는 이 문서의 preview matrix를 정확히 사용한다. 다른 count는 네 조합
`rectangle/horizontal`, `rectangle/vertical`, `square/horizontal`, `square/vertical`을
round-robin으로 최대한 균등하게 배분하며, 나머지는 앞 조합부터 한 장씩 배정한다.

### 10.3 재현성

- module-level global RNG를 사용하지 않고 generator-local RNG를 사용한다.
- 배경 파일 목록은 항상 정렬한 뒤 RNG로 선택한다.
- 같은 인수, 같은 배경 파일 목록, 같은 라이브러리 버전에서는 동일 결과를 생성해야 한다.
- 배경 데이터 내용이나 OpenCV/Pillow 구현 버전이 바뀌면 픽셀 단위 결과가 달라질 수 있음을
  실행 로그에 알린다.

## 11. 검증 및 승인 절차

### 11.1 파일 검증

- PNG 20개와 JSON 20개가 존재해야 한다.
- 모든 PNG와 JSON stem이 일대일로 대응해야 한다.
- 파일 번호가 `0001`부터 `0020`까지 연속이어야 한다.
- 모든 PNG가 1920x1080 RGB로 열려야 한다.
- JSON의 `imagePath`, `imageWidth`, `imageHeight`가 실제 이미지와 일치해야 한다.

### 11.2 레이블 검증

- `shapes`에는 `roi` polygon 하나만 있어야 한다.
- polygon은 서로 다른 네 점을 가져야 한다.
- 점 순서는 TL -> TR -> BR -> BL이어야 한다.
- polygon은 볼록하고 self-intersection이 없어야 한다.
- 모든 점은 이미지 경계 안에 있어야 한다.
- reference quad 면적은 전체 이미지 면적의 30-46% 범위여야 한다.
- projected top/bottom width ratio는 0.84-0.98이어야 한다.
- rotation 생성 목표는 -10 to +10 degrees이며 corner jitter를 포함한 측정값은
  -10.5 to +10.5 degrees 범위여야 한다.
- 패널 중심의 좌우 shift는 이미지 너비의 +-9% 안에 있어야 한다.
- homography의 determinant와 projected homogeneous denominator가 퇴화하지 않아야 한다.

### 11.3 이미지 내용 검증

- 네 OLED/fringe 조합이 각각 5장이어야 한다.
- 각 5장 그룹에 다섯 배경이 한 번씩 포함되어야 한다.
- 각 5장 그룹에 dark, medium, bright 배경이 최소 한 장씩 포함되어야 한다.
- 배경 평균 밝기가 해당 brightness bin 범위에 들어가야 한다.
- 밝은 배경의 99 percentile이 0.98을 초과해 넓게 clipping되지 않아야 한다.
- 배경은 종이가 아닌 연속된 stage 표면으로 보여야 한다.
- OLED 외곽은 top-view rectangle이 아닌 약한 trapezoid로 보여야 한다.
- 20장에 left shift, center, right shift와 양수/음수 rotation이 모두 포함되어야 한다.
- 검은색 종이의 외곽선이나 종이 받침 형태가 생성되지 않아야 한다.
- frequency는 8-36 cycles를 포함하고 각 fringe 방향에 coarse, medium, dense가 있어야 한다.
- deformation은 mild 8장, moderate 8장, severe 4장이어야 한다.
- illumination gradient는 full 16장, weak 4장이어야 한다.
- boundary feather sigma는 1.5-4.0 px이고 median boundary intensity jump는 0.18 이하여야 한다.
- holder layout은 left-right 7장, top-bottom 7장, four sides 6장이어야 한다.
- 접촉하는 한 변당 holder는 1-3개이고 맞은편은 같은 개수여야 한다.
- holder는 OLED 외곽과 접촉하고 서로 겹치지 않으며 active fringe 영역을 가리지 않아야 한다.
- rounded corner radius와 camera-hole diameter가 정의 범위 안이어야 한다.
- 카메라 홀 위치는 top-center 10장, upper-left 10장이어야 한다.
- 홀 가시성은 visible 7장, partial 7장, hidden 6장이어야 한다.
- hidden 표본에도 내부 생성 mask 기준으로 카메라 홀이 실제 존재해야 한다.
- fringe 방향은 이미지 gradient 통계로 horizontal/vertical 정의와 일치해야 한다.

### 11.4 재현성 검증

같은 인수로 두 개의 임시 디렉터리에 생성하고 다음을 비교한다.

- 정렬된 파일명 목록
- JSON 전체 내용
- PNG 파일의 SHA-256 hash

세 항목이 모두 같아야 preview 재현성 검증을 통과한다. 검증용 임시 데이터는
`data/synthetic/preview_03`에 섞지 않는다.

### 11.5 사용자 승인 단계

1. 자동 검증을 모두 통과한 preview 20장만 생성한다.
2. 사용자가 OLED 크기, trapezoid 정도, 좌우 shift, rotation, rounded corner,
   fringe 왜곡, stage 밝기와 조명 분포, 고정 block의 크기와 배치, 카메라 홀
   가시성을 검토한다.
3. 수정 의견이 있으면 파라미터와 preview를 갱신하고 다시 검토한다.
4. 사용자가 승인한 이후에만 최종 condition 이름과 수량을 확정한다.
5. 최종 데이터는 `data/synthetic/{condition}/`에 새로 생성하며 preview를 재사용하거나
   섞지 않는다.

## 12. 제외 범위와 향후 확장

preview 단계에서 제외하는 항목은 다음과 같다.

- scratch, pit, crack, dead pixel과 같은 표면 또는 표시 결함
- 고정 block에 의한 active fringe 가림과 OLED 부분 유실
- 여러 개의 camera hole 또는 notch
- color fringe와 orthogonal composite fringe
- LabelMe mask, camera-hole shape 및 별도 metadata shape
- `gt_corners.csv` 변환과 모델 학습 연결

향후 실측 이미지와 preview를 비교한 뒤 필요하면 조건별 폴더를 추가한다. 예를 들어
노출, 배경, fringe 방향, 패널 비율, 카메라 홀 위치, blur 수준을 condition으로 분리할 수
있다. 조건명은 실제 생성 파라미터가 확정된 후 정하며, 현재 문서에서는 preview 외의
폴더명을 고정하지 않는다.

## 13. 확정된 가정

- 향후 실측 데이터도 이미지와 같은 stem의 LabelMe JSON을 같은 폴더에 저장한다.
- 실측 및 합성 레이블 이름은 모두 `roi`이다.
- LabelMe polygon은 rounded contour가 아니라 네 직선 변의 가상 교점을 나타낸다.
- 카메라 홀은 영상에는 항상 존재하지만 별도 레이블 대상은 아니다.
- `pmd_system_stage.jpg`의 stage 배치를 실측 장비 환경의 기준으로 사용한다.
- stage 위의 검은색 종이는 실측 환경에 없으므로 합성하지 않는다.
- OLED는 left-right, top-bottom 또는 four sides holder로 고정하며 한 변당 1-3개를 사용한다.
- 카메라는 수직 top-view가 아니라 tilt된 시점에서 촬영한다.
- `preview_03`의 geometry는 `preview_02`와 같고 패널 선형 크기는 `preview_01`의 약 80%이다.
- `preview_03` 승인 전까지 README의 F2는 유지하되 현 preview는 면적 50% 이상
  제약의 예외로 관리한다.
- active fringe 가림 범위가 확정되기 전까지 README의 F8은 유지한다.
- 최종 condition과 수량은 preview 승인 이후 결정한다.
- `preview_01`과 `preview_02`의 기존 산출물과 생성 경로는 변경하지 않는다.
