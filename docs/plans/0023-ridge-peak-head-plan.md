# 0023 ridge peak head plan

이 문서는 `ridge` model에 인접 채널 곱을 사용하는 peak 기반 corner 복원 postprocess를 새 head
`peakprod`로 추가하는 작업을 기록한다.

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `src/models/ridge/postprocessor.py`, `src/models/ridge/preprocessor.py`, `src/models/ridge/wrapper.py`, `docs/models/03-dense-prediction.md` |
| 관련 문서 | [0022-offset-model-plan.md](0022-offset-model-plan.md) |

## 1. 목적과 배경

`ridge` model은 image에서 `(B, 4, Hd, Wd)` edge ridge map을 예측한다. 채널 i는 corner i와
corner (i+1)%4를 지나는 무한 직선을 따라 Gaussian 능선을 만든다. 현재 corner 복원은
`RidgePostprocessor` 하나뿐이며, 각 채널을 weighted point cloud로 보고 PCA로 직선을 fit한 뒤
인접 직선의 교점으로 corner 4개를 구한다.

ridge map에서 `peak` model처럼 봉우리를 잡아 corner를 구하려는 요구가 있었다. 4채널을 1채널로
합쳐 단일 argmax하는 방식은 기하학적으로 취약하다. ridge map은 point blob이 아니라 image를
가로지르는 직선이고, 합치면 corner가 두 직선의 교차점이라 사각형 wireframe이 되며, 단일 argmax는
한 점만 주므로 corner 4개를 얻으려면 불안정한 local-maxima와 NMS가 필요하다.

견고한 대안은 채널을 합치지 않고 corner i가 놓인 두 line map(채널 (i-1)%4와 채널 i)을 곱해
corner별 국소 peak map 4개를 만들고 각각 argmax하는 것이다. 두 line map의 곱은 교점에서 급격히
peak를 이루므로 NMS 없이 4개 봉우리가 안정적으로 잡힌다. 이 방식은 model, preprocessor, loss를
바꾸지 않고 postprocessor만 다르므로, 기존 `--head` 선택 패턴에 새 head 값을 추가하는 최소 변경으로
노출한다.

## 2. 범위

포함 항목은 다음과 같다.

- 새 postprocessor `RidgePeakProductPostprocessor`. 인접 ridge 채널 곱으로 corner별 peak map 4개를
  만들고 argmax로 `(B, 4, 2)` normalized corner를 반환한다.
- `RidgeWrapper`가 `head` 값에 따라 postprocessor를 선택하도록 확장한다. `ridge`는 기존
  line-intersection, `peakprod`는 새 postprocessor를 사용한다. model, preprocessor, loss, metric은
  공유한다.
- canonical 문서 `docs/models/03-dense-prediction.md`에 곱-기반 peak head 절과 head 지원 표기를
  추가한다.

제외 항목은 다음과 같다.

- 4채널 sum 후 단일 peak나 NMS 방식은 구현하지 않는다. 기하학적으로 불안정하다고 판단해 제외한다.
- ridge model, preprocessor, loss, metric은 수정하지 않는다.
- 별도 신규 model package로 분리하지 않는다. ridge의 head 변형으로 유지한다.
- slides 반영은 이번 범위에서 제외한다.
- 현재 repository 내부 파일만 변경한다.

## 3. 설계

새 postprocessor는 기존 `RidgePostprocessor`가 있는 같은 파일에 추가한다. 핵심 흐름은 다음과 같다.

```text
probs = sigmoid(raw_output)          # (N, 4, H, W), channel i = line(i, i+1)
edge_prev = probs.roll(1, dims=1)    # channel i now holds line (i-1)%4
corner_maps = probs * edge_prev      # channel i = corner i peak
idx = corner_maps.reshape(N, 4, H*W).argmax(dim=2)
y = (idx // W) / max(H-1, 1)
x = (idx %  W) / max(W-1, 1)
return stack([x, y], dim=2)          # (N, 4, 2)
```

`PeakPostprocessor`의 argmax 후 정규화 부분을 따르되 입력을 raw logit이 아니라 인접 채널 곱
map으로 바꾼다. `roll(1, dims=1)`로 채널 i 위치에 line (i-1)%4를 정렬하는 방식은 기존
`RidgePostprocessor._intersect_adjacent_lines`가 `points.roll(1, dims=1)`로 인접 직선을 짝짓는
채널 매핑과 일치한다.

곱 전 per-channel 상대 threshold는 우선 적용하지 않는다. 곱한 뒤 각 corner map의 argmax만 취하므로
배경 억제 없이도 교점이 최대가 되는 것이 일반적이다. smoke 검증에서 corner가 degenerate하게 한 점으로
몰리면 곱 전 per-channel `rel_thresh`를 추가한다.

`RidgeWrapper.__init__`은 head 검증과 postprocessor 선택을 다음과 같이 확장한다.

```text
if head not in (None, "ridge", "peakprod"):
    raise ValueError("Unknown ridge head: %s. Supported: ridge, peakprod" % head)
if postprocessor is None:
    if head == "peakprod":
        postprocessor = RidgePeakProductPostprocessor()
    else:
        postprocessor = RidgePostprocessor()
```

preprocessor, loss, metric, optimizer, scheduler 구성은 그대로 둔다. model은 head와 무관하게 동일한
`RidgeModel`이다. `--head`는 이미 `scripts/config.py`의 `get_wrapper_kwargs`가 wrapper로 전달하고
`get_model_name`이 head를 output path segment에 포함하므로 `outputs/.../ridge/<network>_peakprod/`
경로가 기존 ridge 산출물과 충돌하지 않는다. config나 factory 수정은 필요 없다.

## 4. 완료 기준

다음을 모두 충족하면 이 plan을 Done으로 본다.

- `RidgePeakProductPostprocessor`가 `src/models/ridge/postprocessor.py`에 추가되고 import된다.
- `--model ridge --head peakprod`가 새 postprocessor로 dispatch되고, `--head ridge`는 기존
  `RidgePostprocessor`를 유지한다.
- `docs/models/03-dense-prediction.md`가 peakprod head를 반영한다.
- 아래 검증이 통과한다.

## 5. 검증

conda 환경 `pytorch_env`를 활성화하고 ver3 root에서 실행한다. import 확인, head dispatch 확인,
`--head peakprod`와 `--head ridge` 두 head의 smoke train run을 각각 수행한다. loss와 IoU가 finite이고
각 output 디렉토리에 `history.json`, `model.pth`, `run.log`가 생성되면 통과로 본다. peakprod의
곱-map argmax가 corner를 degenerate하게 반환하면 곱 전 per-channel threshold를 추가하고 재검증한다.
검증 통과 후 이 plan의 상태를 Done으로 갱신한다.

## 6. 후속 수정: ridge focal loss positive 결손

peakprod head 도입 후 실제 학습에서 `focal` loss는 급감하는데 `iou`가 정체하는 현상이 확인되었다.
원인 분석 결과 `ridge` model의 focal loss positive anchor가 원천적으로 존재하지 않는 버그였다.

`HeatmapFocalLoss`(`src/components/losses.py`)는 CornerNet 계열 penalty-reduced focal loss로,
positive를 `target.eq(1.0)`인 픽셀로 정의하고 그 개수 `num_pos`로 전체 loss를 정규화한다. `peak`
model은 `PeakPreprocessor`가 채널별 max로 나눠 target 최댓값을 정확히 1.0으로 만들기 때문에 이 조건을
만족한다. 반면 `RidgePreprocessor`는 픽셀-직선 수직거리의 Gaussian을 정규화 없이 사용하므로, 이산
격자에서 거리가 정확히 0인 픽셀이 거의 없어 target 최댓값이 0.99999에서 멈춘다. 그 결과 모든 batch에서
`num_pos == 0`이 되어 positive 항이 사라지고, loss는 배경 억제 항만 남는다.

이 상태에서 focal loss는 다음처럼 잘못된 방향으로 최적화된다. 측정값(56x56 map, 전형적 사각형)은
다음과 같다.

| 예측 | 수정 전 focal | 수정 후 focal |
| --- | --- | --- |
| 전 픽셀 배경 출력 | 0.0000004 (최소) | 7.99 |
| ridge map 완벽 재현 | 8.39 | 0.037 (최소) |

수정 전에는 배경을 비울수록 loss가 작아지고 ridge를 맞히면 오히려 커진다. focal 감소는 corner 품질과
무관한 배경 억제의 진행일 뿐이라 iou가 따라 오르지 않는다.

수정은 `RidgePreprocessor`에서 crest band를 정확히 1.0으로 snap하는 것이다. 각 채널의 직선에서 수직거리
`distance`의 절댓값이 0.5 픽셀 이하인 픽셀을 positive로 보고 target을 1.0으로 설정한다. 이렇게 하면
채널당 dense한 positive line이 생겨 `num_pos > 0`이 되고, focal loss가 ridge를 맞히는 방향으로
최적화된다. shared `HeatmapFocalLoss`와 `peak`는 수정하지 않는다.

### 수정 범위

- `src/models/ridge/preprocessor.py`: Gaussian 계산 후 `distance.abs() <= 0.5`인 픽셀을 1.0으로
  설정한다. `peakprod`와 `ridge` 두 head 모두 같은 target을 공유하므로 두 head에 함께 적용된다.

### 수정 검증

수정 전후로 다음을 확인한다.

- unit 확인: 정규화된 target에서 채널별 `target.eq(1.0)` 픽셀 수가 0보다 크다.
- loss 방향 확인: ridge map 완벽 재현 시 focal loss가 전 픽셀 배경 출력보다 작다.
- smoke 또는 짧은 학습: 동일 setting에서 epoch 진행에 따라 iou가 상승한다.
- 회귀 확인: `--head ridge`와 `--head peakprod` 모두 finite loss와 iou로 정상 동작한다.
