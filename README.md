# MaskFormer-Segmentation for EBHI-seg dataset

## Introduction
This project is about MaskFormer segmentation for EHBI-seg dataset, which detect area of six stage tumor colorectal cancer differentiation

Colorectal cancer which is a common fatal malignancy, the fourth most common cancer in men, and the third most common cancer in women worldwide. The EBHI-Seg dataset contains 5,710 electron microscopic images of histopathological colorectal cancer sections that encompass six tumor differentiation stages: normal, polyp, low-grade intraepithelial neoplasia, high-grade intraepithelial neoplasia, serrated adenoma, and adenocarcinoma. 
Read more about EHBI-seg datasheet in [here](https://arxiv.org/pdf/2212.00532v3.pdf)

In this project I using MaskFormer model for semantic segmentation, read more about model: [paper](https://arxiv.org/pdf/2107.06278.pdf) and [code](https://github.com/facebookresearch/MaskFormer) 

## Getting started

See [Getting started with MaskFormer](https://github.com/facebookresearch/MaskFormer/blob/main/GETTING_STARTED.md)

### Requirements
- Python ≥ 3.6
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.

## Segmentation Project

### Datapath
You can edit project and training on [kaggle version](https://www.kaggle.com/code/maitng/ebhi-segmentation), remind to change datapath in module `Set-up environment`

```# change file path for your work space
data,df = read_data("/kaggle/input/d/maitng/ebhi-seg/EBHI-SEG")
```

### Dataset
- Add more [albumentation](https://github.com/albumentations-team/albumentations#spatial-level-transforms) for `train_dataset` in this module
```
import albumentations as A

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

#add more albumentation
train_transform = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
```

### Dataloader
 - Change `batch_size` to match with your GPU, in here I chose `batch_size=12` for GPU NVIDIA P100
```
#change batch_size 
train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
```

### Training model 
- Install MaskFormer pretrained model
```
from transformers import MaskFormerForInstanceSegmentation

# Replace the head of the pre-trained model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)
```
- You can change another optimizer from [torch.optim](https://pytorch.org/docs/stable/optim.html)
```
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
```
- I choose `epochs=20`, you can easily edit it
```
#change epochs in here
epochs = 20
```
