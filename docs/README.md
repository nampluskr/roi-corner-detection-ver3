# 문서 색인

이 문서는 `roi-corner-detection-ver3`의 canonical 문서 위치와 책임을 안내한다. 구현의 실제 동작이
최우선 기준이며, root README와 아래 architecture, guide, model, reference 문서가 이를 설명한다.

## Canonical Documents

프로젝트를 이해하고 실행하는 데 사용하는 문서는 다음과 같다.

| 문서 | 책임 |
| --- | --- |
| [README.md](../README.md) | 프로젝트 개요, model registry, 기본 실행 기준 |
| [architecture/model-assembly.md](architecture/model-assembly.md) | `model`, `network`, `head` 조립 규칙 |
| [architecture/model-contract.md](architecture/model-contract.md) | image, corner, raw output, wrapper contract |
| [architecture/src-layout.md](architecture/src-layout.md) | `src/` module 책임 경계 |
| [architecture/training-and-inference-flow.md](architecture/training-and-inference-flow.md) | training, evaluation, prediction 흐름 |
| [guides/](guides/) | CLI, data, output, experiment workflow |
| [models/](models/) | model별 표현, target, loss, postprocess |
| [reference/](reference/) | shared loss와 metric 기준 |
| [glossary.md](glossary.md) | project 용어 정의 |

## Implementation Baseline

현재 구현은 `scripts/`, `src/components/`, `src/core/`, `src/data/`, `src/models/`, `src/utils/`를
기준으로 한다. model 선택자는 `reg`, `seg`, `det`, `peak`, `ridge`, `gcn`, `hybrid`, `torchseg`,
`torchdet`, `yolo`, `detr`이다.

실행 script는 `--model`로 model을 선택하고, architecture 또는 external whole-model 이름은
`--network` 또는 `--net`으로 지정한다. model별 세부 head는 `--head`로 지정하며, 기본 output 경로는
다음 규칙을 따른다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

## Documentation Workflow

새 canonical 문서, implementation, 구조 변경은 작업 전에 `docs/plans/NNNN-topic-plan.md`에 범위와
완료 기준을 기록하고 승인을 받는다. 문서와 구현이 충돌하면 현재 구현을 확인한 뒤 canonical 문서를
같은 작업에서 갱신한다.

완료된 plan은 이력으로 보존한다. `plans/0001`부터 `plans/0014`까지는 재구성과 구현 조정 이력이고,
[0015-canonical-documentation-plan.md](plans/0015-canonical-documentation-plan.md)는 현재 문서 체계를
확립한 이력이다.

현재 문서 구조는 다음과 같다.

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
