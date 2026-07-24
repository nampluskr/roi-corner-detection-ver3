# Slide 자료 폴더

이 폴더는 roi-corner-detection 프로젝트 진행 보고 슬라이드의 초안과 자산을 보관한다. 최종
산출물은 pptx 형식이지만 현재 단계에서는 검토를 쉽게 하기 위해 markdown 초안, mermaid 블록
다이어그램, 후처리 시각화 이미지를 사용한다.

폴더 구성은 다음과 같다. 폴더를 먼저 알파벳순으로 표시한다.

```text
slides/
├── assets/
│   ├── make_postprocess_figs.py
│   ├── make_public_dataset_figs.py
│   ├── make_synthetic_variation_figs.py
│   ├── make_transform_figs.py
│   ├── postprocess_det.png
│   ├── postprocess_det_box.png
│   ├── postprocess_det_point.png
│   ├── postprocess_gcn.png
│   ├── postprocess_hybrid.png
│   ├── postprocess_offset.png
│   ├── postprocess_peak.png
│   ├── postprocess_reg_gap.png
│   ├── postprocess_reg_spatial.png
│   ├── postprocess_ridge_pcaline.png
│   ├── postprocess_ridge_peakprod.png
│   ├── postprocess_seg.png
│   ├── public_midv2020_example.png
│   ├── public_smartdoc_example.png
│   ├── synth_background.png
│   ├── synth_camera_hole.png
│   ├── synth_fringe.png
│   ├── synth_holder.png
│   ├── synth_position.png
│   ├── synth_rounding.png
│   ├── transform_offline.png
│   └── transform_online.png
├── outline.md
└── README.md
```

각 파일의 역할은 다음과 같다.

| 파일 | 역할 |
| --- | --- |
| `outline.md` | 페이지별 슬라이드 초안, 블록 다이어그램과 이미지 배치 정의 |
| `assets/make_postprocess_figs.py` | 후처리 시각화 이미지를 재생성하는 스크립트 |
| `assets/make_public_dataset_figs.py` | public dataset 예시 이미지를 재생성하는 스크립트 |
| `assets/make_synthetic_variation_figs.py` | synthetic 합성 변형 변수 예시 이미지를 재생성하는 스크립트 |
| `assets/make_transform_figs.py` | 데이터 변형과 transform 예시 이미지를 재생성하는 스크립트 |
| `assets/postprocess_*.png` | model 계열별 후처리 결과 도식 이미지 |
| `assets/public_midv2020_example.png` | MIDV2020 public dataset corner overlay 예시 |
| `assets/public_smartdoc_example.png` | SmartDoc public dataset corner overlay 예시 |
| `assets/synth_*.png` | synthetic 합성 변형 변수 계열별 예시 이미지 |
| `assets/transform_offline.png` | offline pre-augmentation distortion 변형 예시 |
| `assets/transform_online.png` | dataloader online transform 예시 |

head가 여러 종류인 model은 head별로 후처리 이미지를 나눈다. `reg`는 `postprocess_reg_gap.png`와
`postprocess_reg_spatial.png`, `det`는 `postprocess_det_box.png`와 `postprocess_det_point.png`를 둔다.
`ridge`는 기본 `pcaline` head의 `postprocess_ridge_pcaline.png`에 더해 인접 channel 곱 head의
`postprocess_ridge_peakprod.png`를 둔다. head가 하나인 나머지 model은 계열별 단일 이미지를 사용한다.

후처리 이미지는 실제 학습 checkpoint 산출물이 아니라 개념을 설명하는 도식이다. 정답 corner를 고정한
합성 예시에서 각 model의 raw output 표현과 postprocess 결과를 그렸다. 실제 실험 결과 이미지는 향후
checkpoint로 예측을 수행한 뒤 교체할 수 있다.

transform 이미지는 도식이 아니라 실제 결과다. 합성 fringe panel에 `src/data/transforms.py`의 transform
class를 직접 적용한 before/after 예시이며 corner가 image와 함께 변환된다. offline 이미지는 강한
distortion 계열, online 이미지는 dataloader가 실제로 적용하는 단순 계열을 보인다.

synth 이미지는 synthetic 원본을 합성할 때 사용하는 변형 변수 계열별 예시다. 위치와 자세, corner 라운딩,
외부 지그, 카메라 hole, 배경 밝기, fringe 왜곡 여섯 계열을 각각 한 장으로 그린다. 변수를 실제 값으로
설정해 렌더링한 결과이며 정답 corner가 함께 표시된다. 변수 계열과 범위는 데이터 문서
`docs/architecture/05-data-strategy.md`의 synthetic 합성 변형 변수 절에 정리되어 있다.

public dataset 이미지는 현재 repository의 SmartDoc과 MIDV2020 `gt_corners.csv`를 읽어 생성한 실제
sample 예시다. 각 image는 원본 image 위에 `TL`, `TR`, `BR`, `BL` 순서의 네 corner와
polygon line을 함께 표시한다. 원본 public dataset image가 로컬에 없으면 재생성 script는 실패한다.

이미지를 다시 만들려면 conda 환경 `pytorch_env`를 활성화하고 project root에서 스크립트를 실행한다.

```bash
conda activate pytorch_env
cd <project-root>
python slides/assets/make_postprocess_figs.py
python slides/assets/make_public_dataset_figs.py
python slides/assets/make_transform_figs.py
python slides/assets/make_synthetic_variation_figs.py
```
</content>
