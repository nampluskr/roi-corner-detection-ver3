# Direct Coordinate Regression

`reg`는 image 전체에서 네 corner의 normalized coordinate를 직접 회귀한다. target 생성과 최종 decode가
단순하므로 다른 model을 비교하기 위한 첫 baseline으로 적합하다.

## Structure

`RegModel`은 backbone과 adapter를 `FeatureExtractor`로 조립한 뒤 coordinate head를 적용한다.

```text
image -> extractor -> global or spatial feature -> coordinate head -> (B, 8) logits
```

`gap` head는 global feature에 dropout과 linear projection을 적용한다. `spatial` head는 spatial feature를
projection하고 adaptive pooling과 MLP로 8개 값을 만든다. `RegPostprocessor`는 sigmoid를 적용하고
`(B, 4, 2)`로 reshape한다.

## Training Contract

`RegPreprocessor`는 normalized corner를 변경하지 않는다. 기본 loss는 sigmoid가 적용된 raw output과
corner target 사이의 `WingLoss`이며, wrapper metric은 polygon IoU다.

| CLI | 기본값 | 의미 |
| --- | --- | --- |
| `--model` | `reg` | direct coordinate regression 선택 |
| `--network` | `custom` | backbone 선택 |
| `--head` | `gap` | `gap` 또는 `spatial` head |

직접 회귀는 postprocess 실패 경로가 적지만, local evidence를 명시적으로 보존하지 않는다. corner 근처의
spatial 구조가 성능 병목이면 dense prediction 또는 segmentation 계열과 비교한다.
