# Dataset Format

labeled training, validation, test data는 CSV와 image file의 조합으로 읽는다. `CornerDataset`은 한 행에서
image path와 네 개의 normalized corner를 구성하고, `ImageDataset`은 corner column 없이 image만 읽는다.

## Labeled CSV

corner CSV의 필수 column은 다음과 같다.

```text
image_dir,image_name,x1,y1,x2,y2,x3,y3,x4,y4
```

`image_dir`과 `image_name`은 `os.path.join`으로 결합한다. 좌표는 `[0, 1]` normalized value이며 순서는
`TL`, `TR`, `BR`, `BL`이다. CSV 여러 개를 `--csv_path`에 전달하면 행을 하나의 dataset으로 이어 붙인다.

## Split and Sampling

factory는 seed를 사용해 먼저 train과 remaining split을 `0.6 : 0.4`로 나눈다. remaining split은 valid와
test로 다시 반씩 나뉘므로 기본 비율은 train `0.6`, valid `0.2`, test `0.2`다. training dataloader만
shuffle과 `drop_last`를 사용한다.

`--train_size`, `--valid_size`, `--test_size`는 split 이후의 subset 크기를 제한한다. 이 옵션은 빠른
smoke run에 유용하지만, 비교 실험에서는 같은 seed와 같은 제한값을 유지해야 한다.

## Transform Contract

모든 split은 image를 `--image_size` 정사각형으로 resize하고 tensor 변환과 ImageNet normalization을
적용한다. train split은 horizontal flip, vertical flip, rotation, color jitter, Gaussian blur를 추가한다.

geometric transform은 image와 corner를 함께 변환하고, corner가 image 범위를 벗어나는 rotation은 적용을
건너뛴다. flip은 좌표 변환 뒤에도 `TL`, `TR`, `BR`, `BL` 순서를 보존하도록 corner index를 재배열한다.
