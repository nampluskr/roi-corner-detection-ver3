# Public Dataset Examples Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `docs/architecture/05-data-strategy.md`, `slides/README.md`, `slides/outline.md`, `slides/assets/` |
| 관련 문서 | [05-data-strategy.md](../architecture/05-data-strategy.md), [slides/outline.md](../../slides/outline.md) |

## 목적과 배경

현재 repository에는 public dataset인 SmartDoc과 MIDV2020의 image와 corner label 예시가 준비되어 있다.
canonical data strategy는 public, synthetic, measured 3단계 전략을 설명하지만,
public 단계에서 실제로 어떤 image와 corner label을 사용하는지 보여주는 예시가 부족하다.

이 작업은 SmartDoc과 MIDV2020 sample image 위에 `TL`, `TR`, `BR`, `BL` corner를 표시한 slide asset을
추가하고, docs와 slide outline에서 public 단계의 역할을 더 구체적으로 설명한다.

## 범위

포함 항목은 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| public dataset 설명 | SmartDoc과 MIDV2020이 공통 corner contract로 변환되는 방식을 data strategy 문서에 추가 |
| slide outline | public 단계 예시 image와 설명을 진행 보고 slide 초안에 추가 |
| slide asset | SmartDoc과 MIDV2020의 실제 sample image에 네 corner overlay를 그린 PNG 추가 |
| 재생성 script | 현재 repository의 public `gt_corners.csv`와 원본 image 경로를 읽어 public dataset 예시 PNG를 생성하는 script 추가 |

제외 항목은 다음과 같다.

| 항목 | 제외 이유 |
| --- | --- |
| dataset download 자동화 | public dataset 원본 준비 절차는 현재 작업 범위가 아니다 |
| training code 변경 | 이번 작업은 문서와 발표 asset 보강이다 |
| CSV schema 변경 | 기존 `gt_corners.csv` schema를 그대로 사용한다 |

## 완료 기준

다음 조건을 만족하면 이 plan은 완료된 것으로 본다.

| 기준 | 확인 방법 |
| --- | --- |
| SmartDoc 예시 PNG 생성 | `slides/assets/public_smartdoc_example.png` 존재 확인 |
| MIDV2020 예시 PNG 생성 | `slides/assets/public_midv2020_example.png` 존재 확인 |
| data strategy 문서 갱신 | public 단계 설명에 두 dataset 예시와 asset 경로가 포함됨 |
| slide outline 갱신 | public 단계 slide 설명에 두 예시 image가 삽입됨 |
| slide README 갱신 | 새 script와 asset이 자산 목록과 재생성 명령에 포함됨 |

## 검증

검증은 conda 환경 `pytorch_env`를 활성화한 뒤 project root에서 수행한다.

```bash
python slides/assets/make_public_dataset_figs.py
rtk rg -n "public_smartdoc_example|public_midv2020_example|make_public_dataset_figs" docs slides
rtk git status --short
```
