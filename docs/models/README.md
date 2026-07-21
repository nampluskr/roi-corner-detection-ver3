# Model Guide

이 폴더는 factory가 지원하는 model을 출력 표현과 조립 방식에 따라 설명한다. model 선택은 먼저 corner를
어떤 intermediate representation으로 학습할지 정하고, 그 다음 `--network`와 `--head`를 조합하는
순서로 진행한다.

## Model Map

현재 문서와 model의 대응은 다음과 같다.

| 문서 | model | 표현 |
| --- | --- | --- |
| [reg.md](reg.md) | `reg` | direct coordinate regression |
| [seg.md](seg.md) | `seg` | binary ROI mask |
| [det.md](det.md) | `det` | grid detection |
| [dense-prediction.md](dense-prediction.md) | `peak`, `ridge` | corner dense map |
| [gcn.md](gcn.md) | `gcn` | iterative graph refinement |
| [hybrid.md](hybrid.md) | `hybrid` | learned mask with geometry |
| [external-models.md](external-models.md) | `torchseg`, `torchdet`, `yolo`, `detr` | external whole-model |

모든 model은 최종적으로 `(B, 4, 2)` normalized corner를 반환한다. 공통 입력과 wrapper 경계는
[model-contract.md](../architecture/model-contract.md), 조립 축은
[model-assembly.md](../architecture/model-assembly.md)에서 정의한다.

## Selection Principle

`reg`는 단순하고 빠른 coordinate baseline이다. `seg`와 `hybrid`는 ROI boundary가 안정적인 signal일 때
적합하고, `peak`와 `ridge`는 corner 주변 또는 edge 방향의 spatial evidence를 직접 학습한다. `det`와
external detector는 corner를 class별 pseudo-object로 취급한다. `gcn`은 초기 corner와 반복 refinement가
필요한 경우에 사용한다.

동일한 model 내부 비교에서는 network 또는 head 한 가지만 바꾸고, 서로 다른 model 비교에서는 data
split, image size, output directory rule을 고정한다.
