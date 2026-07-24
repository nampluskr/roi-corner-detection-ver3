---
상태: Done
작성일: 2026-07-22
적용 범위: `docs/architecture/05-data-strategy.md` 신설, `docs/README.md` 색인 갱신, `slides/outline.md` 서론 데이터 페이지 추가, offline pre-augmentation과 online transform 구분 서술 추가
관련 문서: [../architecture/05-data-strategy.md](../architecture/05-data-strategy.md), [../guides/01-dataset-format.md](../guides/01-dataset-format.md), [../README.md](../README.md), [../../slides/outline.md](../../slides/outline.md)
---

## 목적과 배경

현재 canonical 문서는 학습 data의 CSV schema, corner 순서, split, transform을 [Dataset Format
Guide](../guides/01-dataset-format.md)에서 다룬다. 그러나 이 문서는 이미 준비된 labeled data를 어떻게
읽고 검증하는지에 집중하며, data 자체가 왜 public, synthetic, measured 세 단계로 구성되는지, 각 단계의
data 특성이 어떻게 이 project의 설계 제약이 되는지는 설명하지 않는다.

data 3단계 전략과 단계별 특성이 만드는 제약은 model 표현 선택, augmentation 범위, 평가 지표, 실험
순서 전반에 영향을 준다. 따라서 이 내용은 특정 실행 절차가 아니라 project 공통 계약을 설명하는
architecture 계층 문서로 두는 것이 적절하다.

이번 plan은 data 3단계 전략과 F1-F8 제약을 현재 문서 규칙과 구현 용어에 맞게 정리한다.

## 범위

포함 항목은 다음과 같다.

- `docs/architecture/05-data-strategy.md` 신설. public, synthetic, measured 3단계 전략, 단계별 data
  특성, 특성에서 도출되는 project 제약, 합성 data 자동 레이블 원리, 3단계와 현재 `--dataset` logical
  stage의 관계를 서술한다.
- synthetic과 measured image가 소량이므로 학습 전에 offline pre-augmentation으로 표본 수를 늘리는
  단계와, dataloader의 online geometric transform이 서로 다르다는 점을 문서와 slide에 서술한다.
  `src/data/transforms.py`가 제공하는 distortion 계열 transform과 현재 active `get_transform`이 쓰는
  단순 transform을 구분한다.
- `docs/README.md`의 architecture 문서 표와 문서 구조 tree에 새 문서를 추가한다.
- `slides/outline.md` 서론에 데이터 3페이지를 삽입하고 이후 페이지 번호를 순증가로 재조정한다. 데이터
  페이지에 offline pre-augmentation과 online transform 구분을 반영한다.

제외 항목은 다음과 같다.

- 새 source code, dataset loader, 합성 data 생성 script는 만들지 않는다. 현재 구현 범위 밖의 기능은
  일반 이론과 현재 구현을 구분해 서술한다.
- [Dataset Format Guide](../guides/01-dataset-format.md)의 CSV schema, split, transform 실무 내용은
  중복 서술하지 않고 참조로 연결한다.
- 새 slide 이미지나 asset은 만들지 않는다. 서론 데이터 페이지는 mermaid 흐름도와 표로 구성한다.

## 완료 기준

다음을 모두 충족하면 이 plan을 `Done`으로 본다.

- `docs/architecture/05-data-strategy.md`가 3단계 전략, 단계별 특성, 제약 대응, 합성 레이블 원리,
  현재 `--dataset` stage와의 관계를 현재 문서 규칙에 맞게 서술한다.
- `docs/README.md`가 새 문서를 architecture 표와 구조 tree에 반영한다.
- `slides/outline.md`가 서론에 데이터 3페이지를 포함하고 이후 페이지 번호가 연속한다.

## 검증

이번 작업은 문서 생성과 갱신이다. 별도 build나 test는 없으며 다음 항목을 확인한다.

- 새 문서와 갱신 문서가 em dash, 유니코드 화살표, 이모지, 수평 구분선을 사용하지 않고 본문 화살표는
  `$\to$`, code 안 화살표는 ASCII `->`를 사용한다.
- `slides/outline.md`의 페이지 번호가 1부터 연속하며 mermaid node label의 괄호가 인용된다.
- `docs/README.md` 구조 tree에 `05-data-strategy.md`가 포함된다.
