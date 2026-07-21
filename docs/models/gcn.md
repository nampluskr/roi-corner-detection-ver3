# Graph Corner Refinement

`gcn`은 image feature에서 초기 네 corner를 예측한 뒤, corner를 graph vertex로 보고 반복적으로 offset을
정제한다. 모든 refinement step을 supervision에 포함해 초기 예측과 후속 정제가 같은 target으로 수렴한다.

## Structure

`GCNModel`은 global feature에서 initial corner를 만든 뒤 spatial feature에서 vertex feature를 sample한다.
`GCNRefiner`는 네 vertex의 adjacency를 따라 message를 교환하고 offset을 더한다.

```text
image -> extractor -> initial corners -> sampled vertex features -> GCN refinement -> corner sequence
```

raw output은 `(B, T + 1, 4, 2)`이며 첫 step은 initial prediction, 이후 step은 refinement 결과다.
`GCNPostprocessor`는 마지막 step을 standard final corner로 선택한다.

## Training Contract

`GCNPreprocessor`는 standard corner target을 그대로 사용한다. `DeepSupervisedSmoothL1Loss`는 모든 step과
target을 비교하고, 기본 설정은 step에 균등한 가중치를 둔다. wrapper metric은 polygon IoU다.

`--model gcn --network custom --head gcn`이 기본 assembly다. iterations, graph layer 수, offset radius는
model constructor option이며 공통 CLI option은 아니다.

GCN은 direct regression보다 refinement trajectory를 제공하지만, initial prediction과 spatial sampling이
불안정하면 각 step이 같은 오류를 반복할 수 있다. 마지막 step만 보지 말고 학습 loss와 prediction CSV를
함께 확인한다.
