# Backbone Weight Reference

이 문서는 `/mnt/d/backbones`에 있는 가중치 중 current project가 실제로 사용하는 항목만 정리한
reference catalog다. 이 경로에는 여러 실험 workspace가 공유하는 pretrained weight가 함께 보관되어
있으므로, 이 문서는 그 중 이 project의 `model`, `network` 조합이 참조하는 가중치만 다룬다.
architecture와 model registry 기준은
[Model Assembly](../architecture/02-model-assembly.md)를 따른다.

## 1. 사용 원칙

표의 각 가중치는 코드에서 다음처럼 참조된다.

```text
src/components/backbones.py  -> reg, seg, det, peak, ridge, gcn, hybrid의 composable backbone
src/models/torchseg/model.py -> torchseg whole-model network
src/models/torchdet/model.py -> torchdet whole-model network
src/models/yolo/model.py     -> yolo whole-model network
src/models/detr/model.py     -> detr whole-model network
```

가중치 파일은 다음처럼 검증한다. 다운로드 도구와 저장 경로는 환경에 맞게 바꿀 수 있다.

```bash
curl -L --fail --silent --show-error -o <filename> <direct-url>
sha256sum <filename>
```

## 2. Composable backbone

다음 가중치는 `src/components/backbones.py`의 `CustomBackbone`을 제외한 `TorchBackbone`,
`TimmBackbone`이 사용한다. `--network` 또는 `--net`으로 선택하며 `reg`, `seg`, `det`, `peak`, `ridge`,
`gcn`, `hybrid` model 중 backbone을 받는 조합에 연결된다.

| 로컬 파일과 크기 | architecture와 사전학습 | 직접 URL | SHA-256 |
| --- | --- | --- | --- |
| `resnet18-f37072fd.pth`<br>46,830,571 B | ResNet-18, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/resnet18-f37072fd.pth) | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `resnet34-b627a593.pth`<br>87,319,819 B | ResNet-34, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/resnet34-b627a593.pth) | `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` |
| `resnet50-0676ba61.pth`<br>102,530,333 B | ResNet-50, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/resnet50-0676ba61.pth) | `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` |
| `efficientnet_b0_rwightman-7f5810bc.pth`<br>21,444,401 B | EfficientNet-B0, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth) | `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` |
| `efficientnet_b5_lukemelas-1a07897c.pth`<br>122,540,693 B | EfficientNet-B5, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth) | `1a07897c0d357db7981640f6be44a63420f11deb932344a69768b62ebe272946` |
| `mobilenet_v2-7ebf99e0.pth`<br>14,258,573 B | MobileNetV2, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth) | `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f` |
| `mobilenet_v3_small-047dcff4.pth`<br>10,306,551 B | MobileNetV3-Small, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |
| `mobilenet_v3_large-8738ca79.pth`<br>22,139,423 B | MobileNetV3-Large, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth) | `8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04` |
| `vgg16-397923af.pth`<br>553,433,881 B | VGG-16, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/vgg16-397923af.pth) | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| `vgg19-dcbb9e9d.pth`<br>574,673,361 B | VGG-19, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |
| `vit_b_16-c867db91.pth`<br>346,328,529 B | ViT-B/16, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/vit_b_16-c867db91.pth) | `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` |
| `swin_t-704ceda3.pth`<br>113,445,839 B | Swin-T, ImageNet-1K | [PyTorch](https://download.pytorch.org/models/swin_t-704ceda3.pth) | `704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b` |
| `wide_resnet50_2.tv_in1k/model.safetensors`<br>275,835,296 B | Wide-ResNet-50-2, ImageNet-1K, `--network wide_resnet50_2`, timm ID `wide_resnet50_2.tv_in1k` | [Hugging Face](https://huggingface.co/timm/wide_resnet50_2.tv_in1k/resolve/main/model.safetensors) | `df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c` |
| `deit_base_distilled_patch16_224.fb_in1k/model.safetensors`<br>349,367,122 B | DeiT-Base distilled, ImageNet-1K, 224, `--network deit_base_distilled`, timm ID `deit_base_distilled_patch16_224.fb_in1k` | [Hugging Face](https://huggingface.co/timm/deit_base_distilled_patch16_224.fb_in1k/resolve/main/model.safetensors) | `ccc9d1bbeede1fc8609a7a7482773a35057e9f38035b9b804bced9126c5a70dc` |
| `cait_s24_224.fb_dist_in1k/model.safetensors`<br>187,709,078 B | CaiT-S24, ImageNet-1K, 224, `--network cait_s24`, timm ID `cait_s24_224.fb_dist_in1k` | [Hugging Face](https://huggingface.co/timm/cait_s24_224.fb_dist_in1k/resolve/main/model.safetensors) | `ec4c0b0e1851c9b1850709caf5cbbcb0ecf121efb4317eedd99ceb0e0e0f4ddb` |

`resnet18`부터 `swin_t`까지는 `TorchBackbone`이 `torch.load`로 읽고, `wide_resnet50_2`, `deit_base_distilled`,
`cait_s24` alias는 `TimmBackbone`이 `TIMM_MODEL_NAMES`로 원본 timm 식별자를 찾아
`safetensors.torch.load_file`로 읽는다. `custom` network는 pretrained weight가 없는 `CustomBackbone`이며
이 표에 포함하지 않는다.

## 3. External whole-model

다음 가중치는 `torchseg`, `torchdet`, `yolo`, `detr` model이 COCO 사전학습 whole architecture를 불러온
뒤 classifier head를 project의 4-corner 또는 binary mask 출력으로 교체한다. 자세한 head 교체 방식은
[External Models](../models/07-external-models.md)를 참고한다.

| 로컬 파일과 크기 | architecture와 사전학습 | project 적용 | 직접 URL | SHA-256 |
| --- | --- | --- | --- | --- |
| `fcn_resnet50_coco-1167a1af.pth`<br>141,567,418 B | FCN ResNet-50, COCO segmentation | `torchseg`, `network=fcn_resnet50` | [PyTorch](https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth) | `1167a1affa42e1e62858f8d3fac12d109e0108327ffc91c5855a324b11683c36` |
| `deeplabv3_resnet50_coco-cd0a2569.pth`<br>168,312,152 B | DeepLabV3 ResNet-50, COCO segmentation | `torchseg`, `network=deeplabv3_resnet50` | [PyTorch](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) | `cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06` |
| `deeplabv3_mobilenet_v3_large-fc3c493d.pth`<br>44,356,159 B | DeepLabV3 MobileNetV3-Large, COCO segmentation | `torchseg`, `network=deeplabv3_mobilenet_v3_large` | [PyTorch](https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth) | `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70` |
| `lraspp_mobilenet_v3_large-d234d4ea.pth`<br>13,097,061 B | LR-ASPP MobileNetV3-Large, COCO segmentation | `torchseg`, `network=lraspp_mobilenet_v3_large` | [PyTorch](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth) | `d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280` |
| `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`<br>167,502,836 B | Faster R-CNN ResNet-50-FPN, COCO detection | `torchdet`, `network=fasterrcnn_resnet50_fpn`, `label_offset=1` | [PyTorch](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| `retinanet_resnet50_fpn_coco-eeacb38b.pth`<br>136,595,076 B | RetinaNet ResNet-50-FPN, COCO detection | `torchdet`, `network=retinanet_resnet50_fpn`, `label_offset=0` | [PyTorch](https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth) | `eeacb38b7cec8cf93c57867e05eaab621047f19b0d2ec5accaa405f690da15b7` |
| `ssd300_vgg16_coco-b556d3b4.pth`<br>142,594,222 B | SSD300 VGG16, COCO detection | `torchdet`, `network=ssd300_vgg16`, `label_offset=1`, 입력이 내부에서 항상 300x300으로 강제 resize됨 | [PyTorch](https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth) | `b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d` |
| `yolov8n.pt`<br>6,549,796 B | Ultralytics YOLOv8-Nano, COCO detection | `yolo`, `network=yolov8n` | [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `facebook-detr-resnet-50/config.json`<br>4,592 B | Hugging Face `facebook/detr-resnet-50`, COCO detection | `detr`, `network=detr_resnet50` | 검증된 직접 URL 없음 | `e7bcf3992363f27717a863f14b193140ad2e41d4338ee012730e58a92cae17e6` |
| `facebook-detr-resnet-50/model.safetensors`<br>166,587,896 B | Hugging Face `facebook/detr-resnet-50`, COCO detection | `detr`, `network=detr_resnet50` | 검증된 직접 URL 없음 | `830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81` |
| `facebook-detr-resnet-50/preprocessor_config.json`<br>290 B | Hugging Face `facebook/detr-resnet-50`, COCO detection | `detr`, `network=detr_resnet50` | 검증된 직접 URL 없음 | `0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b` |

`detr_resnet50`은 `transformers.DetrForObjectDetection.from_pretrained`가 `local_files_only=True`로
directory 전체를 읽으므로 세 파일이 모두 있어야 한다. URL 열에서 `검증된 직접 URL 없음`은 로컬
directory의 정확한 upstream release를 신뢰성 있게 특정하지 못했다는 뜻이다. 다른 PC에서 이 backbone을
복원할 때는 Hugging Face `facebook/detr-resnet-50` repository의 `config.json`,
`preprocessor_config.json`과 `model.safetensors`를 함께 받고 표의 SHA-256과 비교한다.

## 4. 파일 무결성 검증 결과

표의 byte 크기와 SHA-256은 2026-07-24에 `/mnt/d/backbones`의 로컬 파일을 다시 계산한 결과다.
torchvision과 timm 파일명에 포함된 짧은 hash는 배포 식별자이며, 표의 SHA-256은 전체 파일 검증값이다.
같은 경로의 조건부, 비권장 가중치와 상위 분류 기준은 이 catalog 범위 밖이다.
