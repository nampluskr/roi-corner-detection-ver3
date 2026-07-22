# 문서 색인

이 문서는 current project의 canonical 문서 위치, 권장 읽기 순서와 문서별 책임을 안내한다. 구현의 실제
동작이 최우선 기준이며, root README와 아래 architecture, guide, model, reference, glossary가 현재 구조와
사용 방법을 설명한다.

## 1. 처음 읽는 사용자

Python과 neural network 기초 용어는 알지만 ROI corner detection은 처음이라면 다음 순서를 권장한다.

1. [Glossary](glossary.md)에서 ROI, corner order, target, raw output, loss와 metric을 확인한다.
2. [Model Contract](architecture/01-model-contract.md)에서 image와 corner tensor의 공통 흐름을 이해한다.
3. [Model Assembly](architecture/02-model-assembly.md)에서 `model`, `network`, `head`를 구분한다.
4. [Model Guide](models/README.md)에서 coordinate, mask, dense map, detection, refinement 표현을 비교한다.
5. [Dataset Format](guides/01-dataset-format.md)과 [CLI Usage](guides/02-cli-usage.md)로 첫 실행을 준비한다.
6. [Training Workflow](guides/03-training-workflow.md)로 smoke run, 평가, prediction을 수행한다.
7. [Loss Reference](reference/01-losses.md)와 [Metric Reference](reference/02-metrics.md)로 결과를 해석한다.

용어를 먼저 전부 외우기보다 실행 흐름에서 다시 만날 때 glossary로 돌아오는 방식이 좋다.

## 2. 바로 실행하려는 사용자

project를 먼저 실행하려면 다음 문서만 순서대로 읽을 수 있다.

| 순서 | 문서 | 얻는 결과 |
| ---: | --- | --- |
| 1 | [Root README](../README.md) | project 개요와 quick start command |
| 2 | [Dataset Format](guides/01-dataset-format.md) | labeled CSV와 corner 검증 방법 |
| 3 | [CLI Usage](guides/02-cli-usage.md) | model별 command와 option |
| 4 | [Training Workflow](guides/03-training-workflow.md) | smoke run부터 full experiment 절차 |
| 5 | [Experiment Output](guides/04-experiment-output.md) | history, checkpoint, metric, prediction 해석 |

처음 smoke run은 external weight가 필요 없는 `reg/custom/gap`, image size 224, 작은 subset, worker 0을
권장한다.

## 3. Model을 연구하는 사용자

model 원리와 slide 자료의 source를 찾는 경우 다음 경로가 적합하다.

1. common contract와 model assembly를 읽는다.
2. [Model Guide](models/README.md)에서 표현별 비교표와 선택 질문을 확인한다.
3. 관심 model 문서의 문제 정의, architecture, target, loss, postprocess, failure mode를 읽는다.
4. loss와 metric reference에서 수식과 aggregation을 확인한다.
5. source layout과 code mapping으로 실제 class를 추적한다.

각 상세 문서는 개념, 흐름도, tensor shape, 수식, 비교표, failure diagnosis와 code mapping을 포함하므로
후속 slide 문서의 원자료로 사용할 수 있다. current repository에는 slide 산출물 자체는 포함하지 않는다.

## 4. Architecture 문서

공통 구조와 실행 계약을 설명하는 문서는 다음과 같다.

| 순서 | 문서 | 책임 |
| ---: | --- | --- |
| 1 | [01-model-contract.md](architecture/01-model-contract.md) | image, corner order, target, raw output, final output과 failure contract |
| 2 | [02-model-assembly.md](architecture/02-model-assembly.md) | model, network, head, wrapper와 factory 조립 |
| 3 | [03-src-layout.md](architecture/03-src-layout.md) | source directory 책임, dependency direction과 확장 절차 |
| 4 | [04-training-and-inference-flow.md](architecture/04-training-and-inference-flow.md) | split, training, warmup, early stopping, evaluation, prediction lifecycle |
| 5 | [05-data-strategy.md](architecture/05-data-strategy.md) | public, synthetic, measured 3단계 전략과 단계별 특성이 만드는 project 제약 |

architecture 문서는 특정 model 하나보다 package 사이의 공통 경계를 설명한다.

## 5. Guide 문서

실제 작업 절차를 설명하는 문서는 다음과 같다.

| 순서 | 문서 | 책임 |
| ---: | --- | --- |
| 1 | [01-dataset-format.md](guides/01-dataset-format.md) | CSV schema, normalized corner, split, transform과 dataloader |
| 2 | [02-cli-usage.md](guides/02-cli-usage.md) | parser option, model별 command와 current limitation |
| 3 | [03-training-workflow.md](guides/03-training-workflow.md) | data audit, smoke run, full training과 비교 실험 |
| 4 | [04-experiment-output.md](guides/04-experiment-output.md) | output naming, file schema, overwrite와 결과 해석 |
| 5 | [05-synthetic-generation.md](guides/05-synthetic-generation.md) | fringe image와 LabelMe JSON 생성, 변형 변수, hole 가시성, corner 레이블, gt_corners 변환 |

guide는 command를 나열하는 데 그치지 않고 전제 조건, 생성 결과, 확인 방법과 흔한 오류를 함께 설명한다.

## 6. Model 문서

model 표현별 상세 문서는 다음과 같다.

| 순서 | 문서 | model |
| ---: | --- | --- |
| 1 | [01-reg.md](models/01-reg.md) | `reg` direct coordinate regression |
| 2 | [02-seg.md](models/02-seg.md) | `seg` binary ROI segmentation |
| 3 | [03-dense-prediction.md](models/03-dense-prediction.md) | `peak`, `ridge` dense Gaussian map |
| 4 | [04-det.md](models/04-det.md) | `det` custom grid detection |
| 5 | [05-gcn.md](models/05-gcn.md) | `gcn` iterative graph refinement |
| 6 | [06-hybrid.md](models/06-hybrid.md) | `hybrid` learned mask and geometry |
| 7 | [07-external-models.md](models/07-external-models.md) | `torchseg`, `torchdet`, `yolo`, `detr` whole-model |

[Model Guide](models/README.md)는 위 문서의 진입점이며 표현별 차이와 model 선택 질문을 제공한다.

## 7. Reference와 glossary

반복되는 계산과 용어의 기준 문서는 다음과 같다.

| 순서 | 문서 | 책임 |
| ---: | --- | --- |
| 1 | [01-losses.md](reference/01-losses.md) | logit, reduction, shared loss 수식, native loss와 current edge case |
| 2 | [02-metrics.md](reference/02-metrics.md) | IoU, MCD, MaxCD, PCK, success rate와 invalid aggregation |
| reference | [glossary.md](glossary.md) | geometry, data, model assembly, training, evaluation, output 용어 연결 |

reference는 특정 model 문서의 반복을 줄이고 여러 model에 공통으로 적용되는 계산 기준을 제공한다.

## 8. Implementation baseline

현재 구현은 `scripts/`, `src/components/`, `src/core/`, `src/data/`, `src/models/`, `src/utils/`를 기준으로
한다. model registry는 다음 11개 선택자를 제공한다.

```text
reg, seg, det, peak, ridge, gcn, hybrid, torchseg, torchdet, yolo, detr
```

CLI는 `--model`로 표현 package를, `--network` 또는 `--net`으로 backbone이나 whole-model을, `--head`로
model별 variant를 선택한다. 기본 output path는 다음 규칙이다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

current project 동작과 문서가 충돌하면 source implementation을 확인한 뒤 canonical 문서를 같은 작업에서
갱신한다.

## 9. Current 문서 구조

canonical 문서 구조는 다음과 같다. folder를 먼저 알파벳순으로 표시한다.

```text
docs/
├── architecture/
│   ├── 01-model-contract.md
│   ├── 02-model-assembly.md
│   ├── 03-src-layout.md
│   ├── 04-training-and-inference-flow.md
│   └── 05-data-strategy.md
├── guides/
│   ├── 01-dataset-format.md
│   ├── 02-cli-usage.md
│   ├── 03-training-workflow.md
│   ├── 04-experiment-output.md
│   └── 05-synthetic-generation.md
├── models/
│   ├── 01-reg.md
│   ├── 02-seg.md
│   ├── 03-dense-prediction.md
│   ├── 04-det.md
│   ├── 05-gcn.md
│   ├── 06-hybrid.md
│   ├── 07-external-models.md
│   └── README.md
├── plans/
├── reference/
│   ├── 01-losses.md
│   └── 02-metrics.md
├── glossary.md
└── README.md
```

folder 안의 chapter 문서는 `01`부터 시작하는 독립 numbering을 사용한다. root와 folder entry point,
glossary에는 chapter number를 붙이지 않는다. plan은 별도 `NNNN-` 순증가 규칙을 사용한다.

## 10. 문서별 책임 경계

문서 중복을 줄이기 위해 다음 기준을 사용한다.

| 질문 | 우선 문서 |
| --- | --- |
| project를 어떻게 실행하는가 | root README와 guides |
| tensor와 component가 어떤 계약을 지키는가 | architecture |
| 한 model이 왜 필요하고 어떻게 작동하는가 | models |
| loss와 metric 수식은 무엇인가 | reference |
| 반복 용어의 current 정의는 무엇인가 | glossary |
| 변경 작업은 왜, 어떤 범위로 수행했는가 | plans |

model 문서는 필요한 수식을 요약할 수 있지만 shared 계산의 최종 설명은 reference를 우선한다. guide는
source 설계를 반복하기보다 사용자가 수행할 command와 확인 절차를 우선한다.

## 11. Documentation workflow

canonical 문서, implementation, 구조를 변경하는 작업은 먼저 `docs/plans/NNNN-topic-plan.md`에 범위와
완료 기준을 기록하고 승인을 받는다. 완료된 plan은 삭제하지 않고 history로 보존한다.

plan의 기본 상태는 다음과 같다.

| 상태 | 의미 |
| --- | --- |
| `Draft` | 제안 중이며 아직 실행 승인을 받지 않음 |
| `Approved` | 승인된 범위를 구현하거나 문서화하는 중 |
| `Done` | 완료 기준과 검증을 모두 충족함 |

완료된 reconstruction과 문서 체계 작업은 `plans/`에서 확인할 수 있다. 상세 초보자 문서 개정 범위와
진행 기록은 [0016-beginner-documentation-expansion-plan.md](plans/0016-beginner-documentation-expansion-plan.md)에
보존한다.

## 12. 현재 제약을 읽는 방법

문서에서 일반 이론과 current implementation을 구분한다. `일반적으로`, `가능하다`는 설명이 current CLI에서
즉시 지원된다는 뜻은 아니다. 실제 지원 여부는 다음 항목을 확인한다.

1. source registry에 model과 network가 있는지 확인한다.
2. wrapper constructor가 option을 받는지 확인한다.
3. `get_wrapper_kwargs()`가 CLI option을 전달하는지 확인한다.
4. model 문서의 current limitation과 failure mode를 확인한다.
5. 작은 smoke run으로 target, raw output, final output shape를 검증한다.

## 13. 핵심 요약

초보자는 glossary, model contract, assembly, model guide, 실행 guide, loss와 metric reference 순서로 읽는다.
빠른 실행이 목적이면 root README와 guide 4종을 먼저 따른다. 모든 model은 common corner contract로
평가되지만 target, raw output, loss와 postprocess가 다르다. 문서는 current implementation을 설명하는
canonical baseline이며 변경 이력은 plan에 보존한다.
