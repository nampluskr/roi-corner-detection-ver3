# Dataset Format Guide

이 문서는 학습용 CSV와 image를 준비하고, corner 순서와 좌표 범위를 검증하며, current data pipeline이
train, validation, test sample을 만드는 과정을 설명한다. dataset 오류는 model보다 앞에서 발생하지만
loss가 줄지 않거나 corner가 뒤바뀌는 model 오류처럼 보일 수 있으므로 첫 smoke training 전에 확인해야
한다.

## 1. 준비할 파일

labeled dataset은 image file과 CSV file의 조합이다. CSV는 image 위치와 네 corner 좌표를 연결한다.
project 안에 image를 복사할 필요는 없으며 CSV의 `image_dir`이 실제 directory를 가리키면 된다.

권장 예시 구조는 다음과 같다.

```text
my-dataset/
├── images/
│   ├── sample_0001.jpg
│   ├── sample_0002.jpg
│   └── sample_0003.png
└── gt_corners.csv
```

여러 dataset을 함께 사용할 때는 각 dataset에 CSV를 두고 `--csv_path`에 여러 경로를 전달할 수 있다.

gt_corners.csv를 직접 작성하는 대신 raw 주석에서 변환할 수도 있다. `data/make_gt_corners.py`는
`--dataset` 값으로 `smartdoc`, `midv2020`, `images`, `labelme`를 받아 각 형식을 이 CSV schema로
변환한다. LabelMe 라이브러리로 레이블한 image와 JSON 폴더는 `labelme` parser를 사용하며, 하위 폴더까지
재귀로 탐색한다. 합성 image와 LabelMe JSON을 생성하고 변환하는 전체 절차는 [Synthetic Generation Guide](05-synthetic-generation.md)에서
다룬다.

## 2. Labeled CSV의 필수 column

`CornerDataset`이 요구하는 header는 다음과 같다.

```text
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
```

각 column의 의미는 다음과 같다.

| column | 의미 | 예시 |
| --- | --- | --- |
| `image_dir` | image가 있는 directory | `/data/documents/images` |
| `image_name` | directory 아래 file 이름 | `sample_0001.jpg` |
| `x1`, `y1` | top-left corner | `0.12`, `0.08` |
| `x2`, `y2` | top-right corner | `0.88`, `0.10` |
| `x3`, `y3` | bottom-right corner | `0.91`, `0.90` |
| `x4`, `y4` | bottom-left corner | `0.09`, `0.87` |

한 행의 실제 예시는 다음과 같다.

```csv
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
/data/documents/images,sample_0001.jpg,0.12,0.08,0.88,0.10,0.91,0.90,0.09,0.87
```

CSV reader는 column 이름으로 값을 찾는다. column 순서를 바꾸더라도 이름이 정확하면 읽을 수 있지만,
오탈자나 앞뒤 공백이 있으면 `KeyError`가 발생한다.

## 3. Corner 순서

네 점은 단순한 point 집합이 아니라 고정된 순서를 가진 polygon이다.

```text
1: TL -------- 2: TR
   |                |
   |      ROI       |
   |                |
4: BL -------- 3: BR
```

순서는 `TL`, `TR`, `BR`, `BL`이다. CSV column과 tensor index의 대응은 다음과 같다.

| CSV | tensor index | 이름 |
| --- | ---: | --- |
| `(x1, y1)` | `0` | top-left, `TL` |
| `(x2, y2)` | `1` | top-right, `TR` |
| `(x3, y3)` | `2` | bottom-right, `BR` |
| `(x4, y4)` | `3` | bottom-left, `BL` |

이 순서를 유지하면 네 점을 차례로 연결했을 때 ROI boundary를 한 방향으로 순회한다. `BL`과 `BR`을
바꾸면 polygon edge가 교차할 수 있고 coordinate loss도 서로 다른 위치를 대응시킨다.

## 4. Normalized coordinate

CSV 좌표는 pixel이 아니라 `[0, 1]` normalized coordinate다. image width를 $W$, height를 $H$, pixel
coordinate를 $(u, v)$라고 하면 다음과 같이 변환한다.

$$
x = \frac{u}{W}, \qquad y = \frac{v}{H}
$$

반대로 normalized coordinate를 화면에 표시할 pixel로 바꾸려면 다음 계산을 사용한다.

$$
u = xW, \qquad v = yH
$$

예를 들어 width 1000, height 600 image의 pixel corner `(250, 120)`은 normalized `(0.25, 0.20)`이다.
normalized 값을 쓰면 image가 224 정사각형으로 resize되어도 coordinate label을 다시 변경할 필요가 없다.

현재 구현은 CSV 값이 `[0, 1]`인지 자동으로 clamp하거나 검증하지 않는다. pixel 좌표를 그대로 넣으면
dataset은 읽을 수 있지만 target 생성과 metric이 잘못된다.

## 5. Image path 해석

`CornerDataset`은 다음 방식으로 image path를 만든다.

```python
image_path = os.path.join(row["image_dir"], row["image_name"])
```

따라서 `image_dir`은 CSV file 위치를 기준으로 자동 해석되는 상대 경로가 아니다. 상대 경로를 사용하면
script를 실행한 current working directory를 기준으로 해석된다. 다른 위치에서도 안정적으로 실행하려면
절대 경로를 사용하는 편이 명확하다.

image를 열 때 Pillow의 `convert("RGB")`를 적용한다. grayscale이나 alpha channel image도 model 입력에서는
3-channel RGB가 된다.

## 6. 여러 CSV 결합

`--csv_path`는 하나 이상의 경로를 받는다. 다음 command는 두 CSV의 row를 지정한 순서대로 연결한다.

```bash
python scripts/train.py \
  --csv_path /data/set-a/gt_corners.csv /data/set-b/gt_corners.csv \
  --model reg --network custom --head gap
```

결합 후 하나의 전체 sample list에서 random split을 수행한다. dataset별로 먼저 60:20:20을 만들고 합치는
방식이 아니다. 특정 source dataset의 비율을 split마다 정확히 보장해야 한다면 current factory만으로는
stratified split을 제공하지 않으므로 별도 구현이 필요하다.

## 7. Default CSV와 명시적 CSV

`scripts/config.py`의 default CSV path는 repository 내부 public dataset CSV를 가리킨다. 기본값은
`data/public/smartdoc/gt_corners.csv`와 `data/public/midv2020/gt_corners.csv`다. 다른 dataset으로 실행할
때는 자신의 dataset을 `--csv_path`로 명시한다.

```bash
python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --model reg --network custom --head gap
```

`--data_dir` option도 parser에 있지만 current dataset factory는 이 값을 사용해 `csv_path`를 다시 계산하지
않는다. 실행에 직접 영향을 주는 값은 CSV 각 행의 `image_dir`과 CLI의 `--csv_path`다.

## 8. Split 원리

factory는 전체 dataset을 seed 기반으로 두 번 나눈다.

```text
100% samples
-> 60% train + 40% temporary
-> 20% valid + 20% test
```

동일한 CSV 목록, row 순서, seed를 사용하면 train, valid, test index를 같은 방식으로 재구성한다. CSV에
row를 추가하거나 목록 순서를 바꾸면 같은 seed라도 split 구성은 달라진다.

split 뒤에는 size limit을 적용할 수 있다.

| option | 적용 대상 | 기본값 |
| --- | --- | ---: |
| `--train_size` | 60% train subset | `5000` |
| `--valid_size` | 20% valid subset | `1000` |
| `--test_size` | 20% test subset | `1000` |

기본값은 전체 split을 반드시 모두 쓰는 설정이 아니다. split에 더 많은 sample이 있어도 위 수만큼 다시
seed 기반으로 선택한다. 전체를 사용하려면 current parser가 integer만 받으므로 command line에서 `None`을
전달할 수 없다. default 또는 source configuration 조정 없이 전체 split을 쓰려면 충분히 큰 정수를
지정할 수 있다.

## 9. Train transform

train split에는 다음 transform이 순서대로 적용된다.

| 순서 | transform | 설정 | corner 변경 |
| ---: | --- | --- | --- |
| 1 | resize | square `image_size` | 변경 없음 |
| 2 | horizontal flip | probability 0.5 | x 변환과 순서 재배열 |
| 3 | vertical flip | probability 0.5 | y 변환과 순서 재배열 |
| 4 | rotation | -5도부터 5도 | image와 point 회전 |
| 5 | color jitter | brightness 0.2, contrast 0.2 | 변경 없음 |
| 6 | Gaussian blur | kernel 3, sigma 0.1부터 2.0 | 변경 없음 |
| 7 | tensor conversion | `[0, 255]` image를 float tensor로 변환 | NumPy를 tensor로 변환 |
| 8 | normalization | ImageNet mean과 std | 변경 없음 |

rotation 뒤 corner 하나라도 `[0, 1]` 범위를 벗어나면 image와 corner 모두 원래 상태로 두어 해당 rotation을
건너뛴다. horizontal과 vertical flip은 새 위치가 다시 `TL`, `TR`, `BR`, `BL` 순서가 되도록 index를
바꾼다.

`transforms.py`에는 perspective, scale, affine, noise 같은 class도 있지만 current `get_transform()`에는
포함되지 않는다. source에 class가 있다는 사실과 현재 training에서 활성화된다는 사실을 구분해야 한다.

## 10. Validation과 test transform

validation과 test는 random augmentation을 사용하지 않는다.

```text
resize -> tensor conversion -> ImageNet normalization
```

같은 sample은 반복해서 읽어도 같은 input tensor가 되어야 metric 비교가 가능하다. train에만 augmentation을
사용하는 이유는 학습 다양성을 늘리면서 평가 기준은 고정하기 위해서다.

## 11. Tensor shape

한 labeled sample과 batch의 형태는 다음과 같다.

| 단계 | image shape | corner shape |
| --- | --- | --- |
| CSV load 직후 | PIL RGB image | NumPy `(4, 2)` float32 |
| transform 후 sample | `(3, H, W)` tensor | `(4, 2)` tensor |
| dataloader batch | `(B, 3, H, W)` tensor | `(B, 4, 2)` tensor |

기본 `H`와 `W`는 모두 224다. corner는 resize 전후에 normalized 값이므로 shape와 값의 의미가 유지된다.

## 12. Dataloader 정책

split별 dataloader 동작은 다음과 같다.

| 항목 | train | valid | test |
| --- | --- | --- | --- |
| shuffle | 사용 | 사용하지 않음 | 사용하지 않음 |
| drop last | 사용 | 사용하지 않음 | 사용하지 않음 |
| current CLI workers | 4 | 4 | 4 |
| pin memory | CUDA가 있으면 사용 | CUDA가 있으면 사용 | CUDA가 있으면 사용 |

train의 마지막 incomplete batch는 버린다. 예를 들어 train sample 10개, batch size 4이면 2개 batch의 8개만
한 epoch에서 사용한다. sample이 batch size보다 적으면 train dataloader 길이가 0이 될 수 있으므로 smoke
test size를 정할 때 주의한다.

worker가 1개 이상이면 persistent worker와 prefetch factor 4를 사용한다. dataset 문제를 진단할 때 worker
process가 traceback을 감출 수 있으므로 `--num_workers 0`으로 다시 실행하면 원인을 보기 쉽다.

## 13. 실행 전 수동 검사

학습 전에 최소한 다음 항목을 확인한다.

1. CSV header가 필수 column과 정확히 일치하는지 확인한다.
2. 각 `image_dir`과 `image_name`을 결합한 file이 존재하는지 확인한다.
3. 모든 좌표가 finite numeric value이고 `[0, 1]` 범위인지 확인한다.
4. point 순서가 `TL`, `TR`, `BR`, `BL`인지 image에 그려 확인한다.
5. polygon이 self-intersection 없이 실제 ROI를 감싸는지 확인한다.
6. 여러 CSV를 결합할 때 좌표 convention이 동일한지 확인한다.
7. train split sample 수가 batch size 이상인지 확인한다.

특히 좌표 순서는 숫자 범위 검사만으로 발견할 수 없다. 몇 장을 직접 시각화하는 과정이 필요하다.

## 14. 작은 dataset smoke test

data loading과 한 epoch 실행을 먼저 확인하는 command 예시는 다음과 같다.

```bash
conda activate pytorch_env
cd <project-root>

python scripts/train.py \
  --csv_path /absolute/path/to/gt_corners.csv \
  --model reg --network custom --head gap \
  --image_size 224 --batch_size 2 \
  --train_size 8 --valid_size 4 \
  --max_epochs 1 --patience 1 \
  --num_workers 0
```

이 실행은 model 성능을 판단하기 위한 실험이 아니다. CSV open, image decode, transform, batch shape,
forward, loss와 validation metric이 끝까지 연결되는지 확인한다.

## 15. 흔한 오류와 해결 순서

dataset 관련 대표 오류는 다음과 같다.

| 증상 | 가능한 원인 | 점검 방법 |
| --- | --- | --- |
| `FileNotFoundError` for CSV | `--csv_path`가 잘못됨 | absolute CSV path 확인 |
| `KeyError: image_dir` | header 오탈자 또는 다른 schema | 첫 줄 header 확인 |
| image open error | 결합된 image path 또는 file 손상 | 해당 path를 직접 확인 |
| coordinate가 중앙 밖으로 크게 나감 | pixel 좌표를 normalized로 오인 | min, max 범위 확인 |
| 좌우 또는 상하 prediction이 교환됨 | label 순서 오류 | point index를 image에 표시 |
| train progress가 batch 없이 끝남 | sample 수가 batch size보다 작고 drop last 사용 | size 또는 batch size 조정 |
| worker process traceback | multiprocessing 안에서 dataset 오류 | `--num_workers 0` 재실행 |
| 같은 seed인데 split이 달라짐 | CSV row 또는 CSV 목록 순서 변경 | 입력 목록과 file version 고정 |

## 16. 현재 제약

current data pipeline의 범위를 이해할 때 다음 제한을 고려한다.

- labeled train, evaluation, prediction script는 corner column이 있는 CSV를 사용한다.
- `ImageDataset` class는 존재하지만 current scripts는 `has_corners=False`를 노출하지 않는다.
- split 비율은 script option으로 노출되지 않고 factory 기본 0.6을 사용한다.
- source dataset별 stratification과 group split은 제공하지 않는다.
- CLI seed는 augmentation과 전체 PyTorch 학습의 완전한 재현성을 보장하지 않는다.
- `--image_size`는 dataloader에는 반영되지만 wrapper 내부 image size에는 전달되지 않는다.

특히 마지막 제약 때문에 dense prediction과 detection 계열의 standard CLI 실행은 224를 유지하는 것이
좋다.

## 17. Code mapping

dataset 동작을 확인할 source는 다음과 같다.

| 기능 | source |
| --- | --- |
| CSV load와 path 결합 | `src/data/dataset.py` |
| random split과 subset | `src/data/dataset.py` |
| transform 구현 | `src/data/transforms.py` |
| active transform 조립 | `src/core/factory.py` |
| dataloader policy | `src/data/dataloader.py` |
| CLI default와 option | `scripts/config.py` |

## 18. 핵심 요약

CSV 한 행은 image path와 `TL`, `TR`, `BR`, `BL` 순서의 normalized corner를 정의한다. 모든 CSV row를
합친 뒤 seed 기반으로 60:20:20 split을 만들고, size limit을 적용한다. train은 joint augmentation과
shuffle, drop-last를 사용하고 valid와 test는 deterministic transform을 사용한다. 학습 전에는 path,
좌표 범위, corner 순서와 작은 dataloader 실행을 반드시 확인해야 한다.
