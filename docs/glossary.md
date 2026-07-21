# Glossary

이 문서는 문서와 CLI에서 반복되는 용어를 현재 구현 기준으로 정의한다.

| 용어 | 정의 |
| --- | --- |
| model | corner 표현, wrapper, preprocessor, postprocessor를 묶는 factory 선택자 |
| network | model이 사용하는 backbone 또는 external whole-model 이름 |
| head | model의 raw output 세부 표현 또는 detection target 크기 선택자 |
| raw output | model forward가 반환하고 postprocessor 이전에 존재하는 native tensor or object |
| final corner | postprocessor가 반환하는 `(B, 4, 2)` normalized corner |
| wrapper | device, optimizer, scheduler, loss, metric, step lifecycle을 관리하는 class |
| preprocessor | standard corner target을 model-specific training target으로 바꾸는 component |
| postprocessor | raw output을 standard final corner로 복원하는 component |
| backbone | image에서 feature를 추출하는 encoder network |
| adapter | native backbone feature를 `FeatureBundle` contract로 바꾸는 component |
| decoder | 낮은 해상도 feature를 dense output용 spatial feature로 복원하는 component |
| neck | 여러 encoder stage를 detection feature로 결합하는 component |
| dense map | image grid 위치마다 score 또는 geometry evidence를 갖는 output map |
| pseudo-box | point-like corner를 detection loss에 맞추기 위해 부여하는 fixed-size box |
| data stage | `public`, `synthetic`, `measured`처럼 output 경로에 기록하는 logical dataset category |

`model`과 `network`는 같은 뜻이 아니다. 예를 들어 `--model seg --network custom --head mask`에서
`seg`는 training contract를, `custom`은 encoder 선택을, `mask`는 output 표현을 정한다.
