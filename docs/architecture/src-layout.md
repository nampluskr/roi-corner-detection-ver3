# Source Layout

`src/`는 공유 component, 실행 core, data pipeline, model package를 분리한다. model을 추가할 때는 먼저
기존 책임 경계에서 재사용할 구성요소를 찾고, task-specific 구현만 해당 model package에 둔다.

## Directory Responsibilities

현재 source directory의 책임은 다음과 같다.

| 경로 | 책임 |
| --- | --- |
| `src/components/` | backbone, adapter, feature bundle, block, decoder, neck, head, loss, metric |
| `src/core/` | factory, trainer, evaluator, predictor |
| `src/data/` | CSV dataset, split dataloader, joint transform |
| `src/models/base/` | model, wrapper, preprocessor, postprocessor base contract |
| `src/models/<model>/` | model별 raw output과 target, decode, training configuration |
| `src/utils/` | geometry, image, IO, visualization helper |

`src.components`는 특정 model package를 import하지 않는다. 반대로 model package는 shared component를
absolute import로 사용한다. 이 방향은 common block을 재사용하면서 model 사이의 순환 의존을 피한다.

## Model Package Layout

대부분의 model package는 다음 파일을 가진다.

```text
src/models/<model>/
├── __init__.py
├── model.py
├── postprocessor.py
├── preprocessor.py
└── wrapper.py
```

`model.py`는 inference raw output만 정의한다. `wrapper.py`는 optimizer, scheduler, loss, metric, native
library 호출 규약을 정의한다. `preprocessor.py`와 `postprocessor.py`는 standard corner와 method 표현의
양방향 변환을 담는다. 일부 model은 base preprocessor 또는 postprocessor를 그대로 재사용하므로 파일 수가
다를 수 있다.

## Script Boundary

`scripts/`는 CLI와 experiment configuration의 진입점이다. script는 `src.core.factory`에서 dataloader와
wrapper를 생성하고, training loop 자체는 `Trainer`, evaluation은 `Evaluator`, prediction CSV 생성은
`Predictor`에 위임한다. model 구현에서 argparse 또는 output path를 직접 다루지 않는다.
