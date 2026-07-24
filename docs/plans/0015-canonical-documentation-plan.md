---
상태: Done
작성일: 2026-07-22
완료일: 2026-07-22
적용 범위: `docs/` canonical 문서 체계
관련 문서: [../README.md](../README.md), [../../README.md](../../README.md)
---

## 목적과 배경

현재 project의 1차 재구성 구현은 완료되었으며, 현재 `README.md`와 `docs/README.md`는 프로젝트 현황과
문서 운영 절차를 제공한다. 그러나 model 조립 계약, 데이터 인터페이스, 학습 및 추론 흐름, model별
구현 선택 기준은 아직 개별 canonical 문서로 정리되지 않았다.

이 plan은 현재 구현을 기준으로 문서 체계를 확장한다. 문서는 과거 재구성 이력을 복제하지 않고,
현재 `scripts/`, `src/components/`, `src/core/`, `src/data/`, `src/models/`, `src/utils/`의 책임과 실제
CLI를 독자가 확인할 수 있게 한다.

## 범위

포함 항목은 다음과 같다.

- `docs/architecture/`에 model 조립, 공통 model 계약, 학습 및 추론 흐름, `src/` 책임 경계를 설명하는
  canonical 문서를 작성한다.
- `docs/guides/`에 CLI 사용법, dataset 형식, experiment output, 학습 workflow 문서를 작성한다.
- `docs/models/`에 model 선택 기준과 `reg`, `seg`, `det`, dense prediction, `gcn`, `hybrid`, external
  whole-model 계열의 구현 문서를 작성한다.
- `docs/reference/`에 현재 구현된 loss와 metric 기준 문서를 작성한다.
- `docs/glossary.md`에 `model`, `network`, `head`, wrapper, raw output 등의 용어를 정의한다.
- 루트 `README.md`와 `docs/README.md`를 새 문서 색인과 연결한다.

제외 항목은 다음과 같다.

- `src/`, `scripts/`, data, experiment, output의 동작 변경은 포함하지 않는다.
- 현재 구현되지 않은 model이나 외부 라이브러리의 일반 이론을 canonical 구현 문서로 추가하지 않는다.
- 완료된 `docs/plans/0001`부터 `0014`까지의 본문을 소급 수정하지 않는다.

## 문서 구조

완료 시 문서 구조는 다음과 같다.

```text
docs/
├── architecture/
│   ├── model-assembly.md
│   ├── model-contract.md
│   ├── src-layout.md
│   └── training-and-inference-flow.md
├── guides/
│   ├── cli-usage.md
│   ├── dataset-format.md
│   ├── experiment-output.md
│   └── training-workflow.md
├── models/
│   ├── README.md
│   ├── dense-prediction.md
│   ├── det.md
│   ├── external-models.md
│   ├── gcn.md
│   ├── hybrid.md
│   ├── reg.md
│   └── seg.md
├── plans/
├── reference/
│   ├── losses.md
│   └── metrics.md
├── glossary.md
└── README.md
```

## 완료 기준

이 plan이 `Done`으로 전환되기 위한 조건은 다음과 같다.

- 각 canonical 문서가 현재 구현의 module 경계와 실제 public interface를 기준으로 작성된다.
- 모든 model 선택자 `reg`, `seg`, `det`, `peak`, `ridge`, `gcn`, `hybrid`, `torchseg`, `torchdet`, `yolo`,
  `detr`가 model 문서에서 누락 없이 다뤄진다.
- CLI 문서가 `--model`, `--network` 또는 `--net`, `--head`와 output 경로
  `outputs/<dataset>/<model>/<network_head>/<exp_name>/`를 현재 구현 기준으로 설명한다.
- `docs/README.md`가 새 폴더와 문서의 책임을 색인으로 제공한다.
- 문서 내 링크가 유효하고, 현재 기준 문서에 obsolete `method` CLI 또는 `src/methods` 경로가 남지 않는다.

## 검증

다음 검증을 수행한다.

- `docs/` 아래 Markdown 링크와 문서 색인이 생성된 파일을 정확히 가리키는지 확인한다.
- `rtk rg`로 `--method`, `--backbone`, `src/methods`와 이전 output 경로 표기가 새 canonical 문서에 남지
  않았는지 확인한다.
- `rtk git diff --check`으로 Markdown 공백 오류가 없는지 확인한다.
- 이 plan의 범위는 문서 작성이므로 Python 실행 테스트는 수행하지 않는다.
