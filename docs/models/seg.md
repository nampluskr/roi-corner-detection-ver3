# Segmentation Corner Recovery

`seg`는 quadrilateral ROI를 binary mask로 예측하고, mask geometry에서 네 corner를 복원한다. corner를
개별 점으로 직접 회귀하는 대신 영역 전체의 evidence를 학습한다.

## Structure

`SegModel`은 stage feature를 `UNetDecoder`로 복원한 뒤 `MaskHead`로 하나의 mask logit channel을 만든다.

```text
image -> extractor stages -> UNetDecoder -> MaskHead -> (B, 1, H, W) mask logits
```

`UNetDecoder`는 encoder stage의 해상도를 순차적으로 upsample하고 skip feature를 더한다. decoder는
feature 복원을 맡고, target rasterization과 corner fitting은 포함하지 않는다.

## Training and Decode

`SegPreprocessor`는 normalized quadrilateral을 mask size에 맞춰 polygon fill target으로 바꾼다. 기본
wrapper는 `BCELoss`와 `DiceLoss`를 함께 사용한다. `SegPostprocessor`는 sigmoid probability map을
threshold하고 mask의 geometry를 이용해 standard corner order를 복원한다.

| CLI | 기본값 | 의미 |
| --- | --- | --- |
| `--model` | `seg` | segmentation model 선택 |
| `--network` | `custom` | stage feature를 제공하는 backbone |
| `--head` | `mask` | binary mask output |

mask가 비어 있거나 geometry가 불안정하면 corner 복원이 실패할 수 있다. 평가 시 IoU뿐 아니라
`predictions.csv`의 mask 기반 corner 순서와 중심 fallback 여부를 확인한다.
