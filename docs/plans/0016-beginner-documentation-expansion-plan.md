---
상태: Done
작성일: 2026-07-22
적용 범위: project root `README.md`와 `docs/` canonical 문서 전체
관련 문서: [0015-canonical-documentation-plan.md](0015-canonical-documentation-plan.md), [../README.md](../README.md), [../../README.md](../../README.md)
---

## 목적과 배경

[0015](0015-canonical-documentation-plan.md)는 현재 구현에 대응하는 canonical 문서의 폴더와 파일을
확립했다. 그러나 작성된 본문은 이미 구조와 용어를 알고 있는 사용자가 빠르게 interface를 확인하는
manual 수준이다. architecture 문서는 contract를 짧게 열거하고, model 문서는 raw output과 loss를
요약하지만, 초보자가 왜 이런 표현을 사용하는지, tensor가 각 단계에서 어떻게 바뀌는지, 학습과
추론의 차이가 무엇인지 이해하기에는 설명이 부족하다.

현재 root README와 canonical 문서의 전체 분량은 약 867줄이다. 참고 대상 문서는 공통 개념, 모델별
원리, 수식, 예시, 실패 진단과 부록을 포함해 훨씬 높은 설명 밀도를 갖는다. 이번 plan은 분량 자체를
목표로 삼지 않고, 초보자가 선행 지식 없이 읽어도 개념과 실제 구현을 연결할 수 있는 서술 수준으로
현재 문서를 개정한다.

향후 별도 작업에서 slide 문서를 만들 수 있도록, 각 문서는 하나의 절에서 하나의 핵심 개념을
설명하고 흐름도, 비교 표, 단계별 예시와 절별 요약을 제공한다. 다만 이번 plan에서는 slide 파일을
직접 만들지 않는다.

## 독자와 작성 원칙

주 독자는 Python과 neural network의 기초 용어는 들어 보았지만 ROI corner detection, dense prediction,
segmentation, object detection, GCN, geometric postprocessing의 작동 원리를 아직 모르는 사용자다.

설명은 다음 원칙을 따른다.

- 새로운 용어는 처음 등장할 때 일상적인 설명, project 정의, 실제 tensor 또는 data 예시의 순서로
  소개한다.
- 각 component가 필요한 이유를 먼저 설명한 뒤 class와 file 이름을 연결한다.
- 학습과 추론을 분리해, target 생성에만 쓰이는 단계와 실제 prediction에 쓰이는 단계를 구분한다.
- tensor shape, normalized coordinate, channel의 의미를 단계마다 명시한다.
- 수식은 기호를 먼저 정의하고, 수식 아래에 수식이 의미하는 동작을 문장으로 다시 설명한다.
- 외부 whole-model도 library 이름만 나열하지 않고 project contract로 변환되는 과정을 설명한다.
- 현재 구현되지 않은 기능은 일반 이론과 현재 구현을 명확히 구분해 서술한다.
- 각 문서는 slide로 재구성하기 쉬운 흐름도, 비교 표, 핵심 요약을 포함하되 slide 문체로 축약하지 않는다.

## 범위

포함 항목은 다음과 같다.

- `docs/architecture/02-model-assembly.md`를 model, network, head, wrapper 개념부터 factory dispatch와
  composable model, external whole-model 조립 과정까지 단계적으로 설명한다.
- `docs/architecture/01-model-contract.md`를 image, corner order, normalized coordinate, raw output,
  target, final prediction, failure representation의 공통 계약서로 확장한다.
- `docs/architecture/03-src-layout.md`를 directory별 책임, dependency direction, 새 model 추가 시 file별
  역할과 변경 지점을 이해할 수 있는 구조 설명서로 확장한다.
- `docs/architecture/04-training-and-inference-flow.md`를 train, validation, evaluation, prediction의
  lifecycle과 wrapper warmup, checkpoint, metric 집계를 포함한 실행 흐름으로 확장한다.
- `docs/guides/`의 네 문서를 초보자가 환경 확인부터 data 준비, smoke training, evaluation, prediction,
  결과 해석까지 순서대로 수행할 수 있는 실행 guide로 확장한다.
- `docs/models/README.md`에 출력 표현별 분류, model 선택 질문, 비교 축과 model 문서 읽는 순서를 추가한다.
- `docs/models/`의 model 문서를 문제 정의, 핵심 아이디어, architecture, target 생성, loss, inference,
  postprocess, 장단점, failure mode, code mapping, 실행 예시 순서로 확장한다.
- `docs/reference/01-losses.md`에 loss가 필요한 이유, logit과 probability, reduction, model별 loss의 수식과
  선택 이유, 흔한 오류를 설명한다.
- `docs/reference/02-metrics.md`에 training loss와 metric의 차이, 각 metric의 계산 원리, aggregation,
  NaN과 success rate 해석, 비교 시 주의점을 설명한다.
- `docs/glossary.md`를 개별 단어 사전에서 관련 개념끼리 연결되는 초보자용 용어 안내서로 확장한다.
- root `README.md`와 `docs/README.md`는 상세 본문을 중복하지 않고 학습 순서와 canonical 문서 경로를
  안내하도록 갱신한다.
- 초보자가 file 목록만 보고도 권장 읽기 순서를 알 수 있도록 chapter 성격의 canonical 문서에 폴더별
  `NN-` 접두사를 추가하고 모든 Markdown link를 새 경로로 갱신한다.

제외 항목은 다음과 같다.

- `src/`, `scripts/`, data, experiment, output의 코드와 동작은 변경하지 않는다.
- PowerPoint, Google Slides, Marp, reveal.js 등의 slide 산출물은 만들지 않는다.
- 현재 구현에 없는 model, loss, metric, data stage 실행 기능을 지원 기능처럼 문서화하지 않는다.
- 완료된 `docs/plans/0001`부터 `0015`까지의 본문은 소급 수정하지 않는다.
- 참고 문서의 project history, obsolete CLI, 이전 directory 이름을 새 canonical 문서에 복사하지 않는다.

## 문서 넘버링

번호는 `docs/` 전체에서 이어지는 전역 번호가 아니라 각 폴더 안에서 `01`부터 다시 시작한다. 이 방식은
architecture, guide, model, reference가 서로 독립적인 학습 경로라는 점을 유지하고, 한 폴더에 문서를
추가할 때 다른 폴더의 번호와 link까지 변경되는 문제를 피한다.

적용할 rename은 다음과 같다.

| 현재 경로 | 변경 경로 |
| --- | --- |
| `docs/architecture/model-contract.md` | `docs/architecture/01-model-contract.md` |
| `docs/architecture/model-assembly.md` | `docs/architecture/02-model-assembly.md` |
| `docs/architecture/src-layout.md` | `docs/architecture/03-src-layout.md` |
| `docs/architecture/training-and-inference-flow.md` | `docs/architecture/04-training-and-inference-flow.md` |
| `docs/guides/dataset-format.md` | `docs/guides/01-dataset-format.md` |
| `docs/guides/cli-usage.md` | `docs/guides/02-cli-usage.md` |
| `docs/guides/training-workflow.md` | `docs/guides/03-training-workflow.md` |
| `docs/guides/experiment-output.md` | `docs/guides/04-experiment-output.md` |
| `docs/models/reg.md` | `docs/models/01-reg.md` |
| `docs/models/seg.md` | `docs/models/02-seg.md` |
| `docs/models/dense-prediction.md` | `docs/models/03-dense-prediction.md` |
| `docs/models/det.md` | `docs/models/04-det.md` |
| `docs/models/gcn.md` | `docs/models/05-gcn.md` |
| `docs/models/hybrid.md` | `docs/models/06-hybrid.md` |
| `docs/models/external-models.md` | `docs/models/07-external-models.md` |
| `docs/reference/losses.md` | `docs/reference/01-losses.md` |
| `docs/reference/metrics.md` | `docs/reference/02-metrics.md` |

진입점과 전역 reference에는 번호를 붙이지 않는다. root `README.md`, `docs/README.md`,
`docs/models/README.md`, `docs/glossary.md`는 기존 이름을 유지한다. `docs/plans/`는 이미 순증가하는
`NNNN-` 규칙을 사용하므로 이번 rename 대상에서 제외한다.

## 문서별 표준 구성

architecture와 model 문서는 주제에 맞게 다음 구성을 적용한다.

1. 이 문서가 답하는 질문과 선수 개념을 설명한다.
2. 문제 상황과 해당 구조 또는 model이 필요한 이유를 설명한다.
3. 핵심 용어와 input, target, raw output, final output을 정의한다.
4. 전체 흐름을 ASCII diagram과 단계별 문장으로 설명한다.
5. 각 단계의 tensor shape와 coordinate 의미를 구체적인 예로 설명한다.
6. 학습 단계의 target과 loss 계산 원리를 설명한다.
7. inference와 postprocess의 작동 원리를 설명한다.
8. current code의 class와 file이 각 개념을 어디에서 구현하는지 연결한다.
9. 정상 동작 예시와 대표 failure mode, 점검 순서를 설명한다.
10. 다른 model 또는 component와의 비교 표와 핵심 요약을 제공한다.

guide 문서는 작업 전제, 단계별 명령, 생성 결과, 확인 방법, 흔한 오류 순서로 작성한다. reference 문서는
직관, 기호 정의, 계산 원리, 구현 대응, 해석 예시, 주의점 순서로 작성한다.

## 예상 작업량

이번 plan은 root README와 docs index를 포함해 약 21개 canonical 문서를 개정한다. 예상치는 source code
재확인, 참고 문서 열람, 본문 작성, 문서 사이의 교차 검토와 수정에 필요한 시간을 모두 포함한다.

model 문서를 먼저 작성하는 순서에 따른 예상 작업량은 다음과 같다.

| 작업 | 예상 시간 | 예상 토큰 |
| --- | ---: | ---: |
| model 문서 8종 | 3~4시간 | 70k~120k |
| 공통 용어와 architecture 4종 | 1.5~2.5시간 | 35k~60k |
| guide 4종 | 1.5~2시간 | 30k~50k |
| loss, metric, glossary | 1.5~2시간 | 30k~50k |
| README, index, 전체 검증 | 0.5~1시간 | 15k~25k |
| 합계 | 8~12시간 | 180k~305k |

현재 약 867줄인 root README와 canonical 문서의 전체 분량은 완료 시 약 4,500~6,500줄로 늘어날 것으로
예상한다. 이 가운데 실제 문서 본문은 약 90k~150k tokens로 예상하며, 나머지 token은 source와 참고
문서 확인, 수정, 정합성 검토에 사용된다. 시간과 token은 설명 과정에서 발견되는 구현 차이와 재검토
범위에 따라 달라질 수 있으므로 고정 budget이 아니라 작업 규모를 판단하기 위한 범위로 사용한다.

## 구현 단계

문서 개정은 다음 순서로 진행한다.

1. 본문 상세화 전에 위 표의 canonical 문서를 rename하고 root README, docs index, model index,
   architecture와 model 문서의 link를 새 경로로 갱신한다. 완료된 plan의 역사적 본문은 수정하지 않는다.
2. `models/README.md`와 model 문서 7종을 가장 먼저 상세화한다. 각 model의 문제 정의, input, target, raw
   output, loss, inference, postprocess를 source와 대조하고, 본문에서 필요한 용어를 처음 등장하는
   위치에서 설명한다.
3. model 문서에서 사용한 개념과 tensor 명칭을 기준으로 architecture 문서와 `glossary.md`를 확장하고,
   공통 입출력 계약과 component boundary를 일관되게 정리한다.
4. loss와 metric reference를 model 문서의 수식, target 표현, 평가 결과와 연결해 상세화한다.
5. CLI, dataset, output, training workflow guide를 실제 script 동작과 대조해 상세화한다.
6. root README와 docs index에 초보자용 권장 읽기 순서를 추가한다.
7. 전체 문서의 cross-link, 용어, code symbol, 설명 수준을 검토하고 model 문서와 공통 문서 사이의
   중복 또는 불일치를 조정한다.

작업 결과는 model 문서, architecture와 glossary, loss와 metric 및 guide, README와 전체 검증의 네
묶음으로 점검한다. 각 묶음이 끝날 때 용어와 설명 깊이를 확인해 이후 slide 문서 작성 단계에서 같은
개념을 다시 조사하거나 원문을 대폭 수정하는 일을 줄인다.

## 진행 상황

2026-07-22에 사용자가 이 plan을 승인하고 model 문서를 먼저 작성하도록 요청했다. model 문서 검토 단계
이후 architecture와 guide 문서 작성도 요청했다. 현재 진행 결과는 다음과 같다.

- `docs/models/README.md`와 numbered model 문서 7종을 초보자용 상세 문서로 개정했다.
- model 문서의 `01`부터 `07`까지 numbering과 내부 link를 적용했다.
- architecture 문서 4종을 초보자용 상세 문서로 개정하고 `01`부터 `04`까지 numbering을 적용했다.
- guide 문서 4종을 data 준비, CLI, training workflow, output 해석 순서의 상세 문서로 개정하고 `01`부터
  `04`까지 numbering을 적용했다.
- `docs/reference/01-losses.md`와 `docs/reference/02-metrics.md`를 수식, 구현 edge case, aggregation과 결과
  해석을 포함하는 초보자용 상세 문서로 개정했다.
- `docs/glossary.md`를 geometry, data, assembly, training, evaluation과 output이 서로 연결되는 개념 안내서로
  개정했다.
- root `README.md`에 project contract, model registry, quick start와 current limitation을 반영했다.
- `docs/README.md`에 초보자, 실행 사용자, model 연구자를 위한 읽기 경로와 numbered canonical index를
  반영했다.
- model index와 architecture, guide, reference 사이의 교차 link를 current numbered filename에 맞게
  갱신했다.
- root README와 canonical 문서는 총 약 6,713줄이며, model 원리, architecture, 실행 guide, loss와 metric,
  glossary를 slide 원자료로 사용할 수 있는 설명 수준으로 확장했다.
- local Markdown link의 실제 file 존재, obsolete CLI와 path, 금지 문자, heading depth, trailing whitespace와
  `git diff --check`을 확인했다.
- 문서만 변경했으므로 Python 실행 test는 수행하지 않았다.
- 모든 완료 기준을 충족해 plan 상태를 `Done`으로 전환했다.

## 완료 기준

이 plan이 `Done`으로 전환되기 위한 조건은 다음과 같다.

- 각 architecture 문서가 용어 나열을 넘어 왜 해당 contract와 boundary가 필요한지 설명한다.
- 모든 model 문서가 input, target, raw output, loss, final output의 shape와 의미를 포함한다.
- 모든 model 문서가 학습과 inference 흐름을 분리하고, preprocessor와 postprocessor의 역할을 설명한다.
- 모든 model 문서가 최소 하나의 단계별 예시, 비교 표, failure mode와 code mapping을 포함한다.
- loss와 metric 문서가 초보자가 계산 목적과 결과 숫자의 의미를 이해할 수 있는 직관과 수식을 제공한다.
- guide 문서만 따라가도 labeled CSV 확인, smoke training, checkpoint evaluation, prediction CSV 확인 순서를
  이해할 수 있다.
- 각 상세 문서에 slide 작성 시 재사용할 수 있는 핵심 흐름도, 비교 표 또는 절별 요약이 포함된다.
- 설명은 현재 source code symbol과 runtime behavior에 대응하며, 일반 이론은 현재 구현과 구분된다.
- `README.md`와 `docs/README.md`가 초보자에게 권장 읽기 순서를 제공한다.
- chapter 성격의 canonical 문서가 정의된 폴더별 `NN-` 접두사를 사용하고, 기존 filename을 가리키는
  current-document link가 남지 않는다.
- canonical 문서에 obsolete `--method`, `--backbone`, `src/methods` 또는 이전 output 경로가 남지 않는다.

## 검증

문서 개정 후 다음을 확인한다.

- model factory, wrapper constructor, preprocessor, postprocessor, loss, metric, script option을 source와
  대조해 class 이름, default option, tensor contract를 확인한다.
- `rtk rg`로 model 11종, CLI option, output path, output file 이름이 canonical 문서에 빠짐없이 있는지
  확인한다.
- `rtk rg`로 obsolete CLI와 path, em dash, Unicode arrow, emoji가 새 canonical 문서에 남지 않았는지
  확인한다.
- local Markdown link가 실제 file을 가리키는지 확인하고 `rtk git diff --check`을 수행한다.
- `rtk rg`로 rename 전 filename을 가리키는 link가 root README와 canonical 문서에 남지 않았는지
  확인한다.
- 문서만 변경하므로 Python 실행 테스트는 수행하지 않는다.
