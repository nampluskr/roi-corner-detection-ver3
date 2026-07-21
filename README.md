# roi-corner-detection-ver3

이 프로젝트는 image에서 ROI의 네 corner를 normalized quadrilateral로 예측하는 PyTorch workspace다.
현재 구현은 model별 표현과 공통 training, evaluation, prediction workflow를 제공하며, 상세 설계는
`docs/`의 canonical 문서에서 관리한다.

## Project Structure

현재 상위 구조는 다음과 같다.

```text
docs/
├── architecture/
├── guides/
├── models/
├── plans/
├── reference/
├── glossary.md
└── README.md
scripts/
src/
├── components/
├── core/
├── data/
├── models/
└── utils/
```

현재 `src/models/`에는 다음 model 패키지가 있다.

| model | 역할 |
| --- | --- |
| `reg` | 직접 코너 좌표 회귀 |
| `seg` | ROI 마스크 세그멘테이션 기반 코너 복원 |
| `det` | 커스텀 검출 head 기반 코너 검출 |
| `peak` | 코너별 가우시안 피크 dense map |
| `ridge` | 변별 가우시안 ridge dense map |
| `gcn` | 초기 코너 추정과 그래프 기반 반복 정제 |
| `hybrid` | DL 마스크 예측과 classical CV 후처리 결합 |
| `torchseg` | torchvision segmentation whole-model 계열 |
| `torchdet` | torchvision detection whole-model 계열 |
| `yolo` | YOLO 계열 검출 model |
| `detr` | DETR 계열 검출 model |

## 실행 기준

학습, 평가, 예측 스크립트는 `--model`을 model 선택자로 사용하고, architecture 또는 외부 whole-model
이름은 `--network` 또는 `--net`으로 지정한다. `--head`는 model별 세부 head 옵션으로 유지한다.

기본 실행 형태는 다음과 같다.

```bash
python scripts/train.py --model reg --network custom --head gap
python scripts/train.py --model seg --network custom --head mask
python scripts/train.py --model yolo --network yolov8n --head box
```

실험 산출물 경로는 다음 규칙을 따른다.

```text
outputs/<dataset>/<model>/<network_head>/<exp_name>/
```

## 문서 안내

문서 색인은 [docs/README.md](docs/README.md)에서 제공한다. model 조립과 공통 contract는
[docs/architecture/](docs/architecture/), 실행 방법은 [docs/guides/](docs/guides/), model별 설명은
[docs/models/](docs/models/), loss와 metric은 [docs/reference/](docs/reference/)에서 확인한다. 완료된
작업 계획은 [docs/plans/](docs/plans/)에 이력으로 보존한다.
