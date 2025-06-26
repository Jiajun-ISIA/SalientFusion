# SalientFusion

## What can I find here?

This repository contains all code and implementations used in:

```
SalientFusion: Context-Aware Compositional Zero-Shot Food Recognition
```
accepted to ICANN 2025



### Datasets:

CZSFood-90 and CZSFood-164

* To be released at a later time



### Training:

**A basic sample run using the best parameters would like this**:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py --clip_arch ViT-L-14.pt --dataset_path dataset_root/food172 --save_path save/food172 --yml_path ./config/salientfusion/food172.yml --num_workers 10 --seed 0

```
## Paper
If you find this work useful, please consider citing:
```
@InProceedings{Jiajun2022,
  author       = "Jiajun Song and Xiaoou Liu",
  title        = "SalientFusion: Context-Aware Compositional Zero-Shot Food Recognition",
  booktitle    = "34th International Conference on Artificial Neural Networks",
  year         = "2025",
}
```
