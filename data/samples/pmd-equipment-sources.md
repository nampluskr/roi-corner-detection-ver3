# PMD 장비 및 OLED 지그 참고 이미지 출처

이 문서는 `data/samples/`에 저장된 PMD 장비 및 OLED positioning jig 참고 이미지의
출처와 로컬 파일 정보를 기록한다. 모든 파일은 2026-07-10에 내려받았다.

`data/`는 `.gitignore` 대상이므로 이 문서와 이미지들은 Git으로 추적되지 않는다.

## PMD 장비 이미지

아래 두 이미지는 공개 논문의 장비 사진이다. 합성 이미지에서 screen, camera, stage,
SUT holder와 주변 기구물의 배치를 설계하기 위한 참고 자료로 사용한다.

| 로컬 파일 | 내용 | 크기 | SHA-256 |
|---|---|---:|---|
| `pmd_system_stage.jpg` | Camera, DLP display, stage가 보이는 PD 시스템 | 726x544 | `2277c3dbf04b82bcbc56368a774b84f886b7810ade2ea0b4566f44425552b6b3` |
| `pmd_calibration_setup.jpg` | Calibration mirror holder와 tracker를 포함한 전체 배치 | 2060x1426 | `620fef2ffcf1042bf43a6c0abda39f0fb1c075b45d69233e855c0556cfa7006b` |

### pmd_system_stage.jpg

- 논문: "Phase Deflectometry for Defect Detection of High Reflection Objects"
- Figure: Figure 5, PD defect detection system
- 논문 페이지: https://pmc.ncbi.nlm.nih.gov/articles/PMC9922010/
- Figure 페이지: https://pmc.ncbi.nlm.nih.gov/articles/PMC9922010/figure/sensors-23-01607-f005/
- 원본 이미지: https://cdn.ncbi.nlm.nih.gov/pmc/blobs/3852/9922010/5d63e81def61/sensors-23-01607-g005.jpg
- 이용 조건: Creative Commons Attribution 4.0
- License: https://creativecommons.org/licenses/by/4.0/

### pmd_calibration_setup.jpg

- 논문: "Laser-tracker-based reference measurement for geometric calibration of
  phase-measuring deflectometry with active display registration"
- Figure: Figure 4, calibration setup
- 논문 페이지: https://jsss.copernicus.org/articles/13/1/2024/
- 원본 이미지: https://jsss.copernicus.org/articles/13/1/2024/jsss-13-1-2024-f04.jpg
- 이용 조건: Creative Commons Attribution 4.0
- License: https://creativecommons.org/licenses/by/4.0/

## OLED Positioning Jig 이미지

아래 두 이미지는 PMD 장비 사진이 아니라 상용 mobile phone positioning jig 사진이다.
OLED를 좌우 또는 상하에서 고정하는 block과 clamp의 배치만 참고한다.

| 로컬 파일 | 내용 | 크기 | SHA-256 |
|---|---|---:|---|
| `oled_jig_loaded.webp` | 스마트폰이 장착된 4방향 positioning jig | 800x800 | `911d1a0ad012fd8c24639ae0e29d96c7bb8e4d47fe2830a5a3b29a528f37484b` |
| `oled_jig_empty.webp` | 네 방향 clamp가 보이는 빈 positioning jig | 800x800 | `bfc5e4e7e5765e6d1373f648eab5ba95aa77bbedb9e2345c20c7257d5109f651` |

### oled_jig_loaded.webp

- 상품 페이지: https://www.ebay.com/itm/387563840286
- 원본 이미지: https://i.ebayimg.com/images/g/utEAAOSwVLBnI~3x/s-l1600.webp
- 이용 조건: 상용 상품 이미지, 공개 라이선스 확인 안 됨
- 사용 제한: 내부 구조 검토를 위한 로컬 참고 자료로만 사용하고 재배포하지 않는다.

### oled_jig_empty.webp

- 상품 페이지: https://www.ebay.com/itm/387563840286
- 원본 이미지: https://i.ebayimg.com/images/g/6SsAAOSw~xJnI~3z/s-l1600.webp
- 이용 조건: 상용 상품 이미지, 공개 라이선스 확인 안 됨
- 사용 제한: 내부 구조 검토를 위한 로컬 참고 자료로만 사용하고 재배포하지 않는다.

## 사용자 확인 결과

사용자가 `pmd_system_stage.jpg`를 실제 측정 환경과 가장 유사한 참고 이미지로
선택했다. 확인된 환경은 다음과 같다.

- OLED 패널은 평면 stage 위에 놓인다.
- 참고 사진의 검은색 종이는 실제 환경에 없다.
- OLED의 맞은편 두 변을 jig 또는 block으로 고정한다.
- 고정 방향은 left-right 또는 top-bottom이다.

고정 block이 bezel만 접촉하는지 active fringe 영역을 일부 가리는지는 아직
확정되지 않았다. 이 항목을 확정하기 전에는 README의 F8을 변경하지 않는다.
