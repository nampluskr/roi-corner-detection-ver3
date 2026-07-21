# Custom Corner Detection

`det`는 네 corner를 서로 다른 class의 pseudo-object로 표현하는 custom grid detector다. 각 grid cell은
corner class score와 class-agnostic box 또는 point regression 값을 만든다.

## Structure

`DetModel`은 encoder stage feature를 neck으로 결합하고 `DetectionHead`를 적용한다.

```text
image -> extractor stages -> detection neck -> DetectionHead -> cls map and box map
```

classification map은 corner class별 objectness를, regression map은 cell 내부 center offset과 선택적으로
box width and height를 표현한다. `DetPostprocessor`는 class마다 가장 높은 score cell을 선택해 normalized
center coordinate를 복원한다.

## Target and Loss

`DetPreprocessor`는 각 corner가 속한 grid cell을 positive로 표시하고 `DetTarget`에 class target, box or
point target, positive mask를 만든다. 기본 loss는 sparse classification을 위한 `FocalLoss`와 positive
cell regression을 위한 masked `SmoothL1Loss`다.

| head | regression channel | target 의미 |
| --- | --- | --- |
| `box` | center offset, width, height | 고정 크기 pseudo-box |
| `point` | center offset | point-like pseudo-box |

`box`의 width와 height는 실제 corner 크기가 아니라 training assignment를 위한 표현이다. 최종 corner는
두 head 모두 선택된 cell의 center와 offset에서 계산한다.

## CLI

기본 실행은 다음과 같다.

```bash
python scripts/train.py --model det --network custom --head box --save
```

grid stride와 neck channel은 wrapper constructor option이지만 현재 공통 CLI에는 노출하지 않는다. CLI에서
지원하는 assembly 축은 model, network, head다.
