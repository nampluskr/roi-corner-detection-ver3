# Hybrid Mask and Geometry

`hybrid`는 learned binary mask와 classical geometry postprocess를 결합한다. mask는 ROI 영역을 제공하고,
postprocessor는 contour approximation, line fitting, intersection, probability map refinement를 사용해
corner를 복원한다.

## Structure

neural network 부분은 `seg`와 유사하게 extractor, `UNetDecoder`, `MaskHead`로 구성된다. 기본 network는
`mobilenet_v3_large`이며, 결과 mask의 geometry 복원은 `HybridPostprocessor`가 담당한다.

```text
image -> mask logits -> sigmoid mask -> contour or lines -> four corners
```

postprocessor는 contour 기반 quadrilateral approximation을 우선 사용하고, 필요하면 side grouping과 line
intersection으로 복원한다. probability map은 최종 corner refinement에 사용된다.

## Training Contract

`HybridPreprocessor`는 `SegPreprocessor`의 polygon mask target을 재사용한다. 기본 loss는 BCE와 soft Dice를
합친 `BCEDiceLoss`이고, wrapper는 polygon IoU와 success rate를 기록한다.

| CLI | 기본값 | 의미 |
| --- | --- | --- |
| `--model` | `hybrid` | hybrid assembly 선택 |
| `--network` | `mobilenet_v3_large` | backbone 선택 |
| `--head` | `hybrid` | learned mask와 geometry decode |

이 model은 mask quality와 geometry rule 모두에 의존한다. metric이 낮을 때는 mask가 불량한지, contour
추출이 실패했는지, line intersection이 불안정한지를 prediction 결과와 함께 분리해 확인한다.
