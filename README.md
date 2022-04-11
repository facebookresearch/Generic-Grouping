# Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity

## Pytorch implementation for "Open-World Instance Segmentation:
Exploiting Pseudo Ground Truth From Learned Pairwise Affinity" ([CVPR 2022, link TBD]()) <br/>

[Weiyao Wang](https://mcahny.github.io/), [Matt Feiszli](https://scholar.google.com/citations?user=_BPdgV0AAAAJ), [Heng Wang](https://scholar.google.co.kr/citations?user=nkmDOPgAAAAJ), [Jitendra Malik](https://rcv.kaist.ac.kr), and [Du Tran](https://weichengkuo.github.io/).

```bibtex
@article{kim2021oln,
  title={Learning Open-World Object Proposals without Learning to Classify},
  author={Kim, Dahun and Lin, Tsung-Yi and Angelova, Anelia and Kweon, In So and Kuo, Weicheng},
  journal={CVPR},
  year={2022}
}
```


## Introduction

Humans can recognize novel objects in this image despite having never seen them  before. “Is it possible to learn open-world (novel) object proposals?” In this paper we propose **Object Localization Network (OLN)** that learns localization cues instead of foreground vs background classification. Only trained on COCO, OLN is able to propose many novel objects (top) missed by Mask R-CNN (bottom) on an out-of-sample frame in an ego-centric video.

<img src="./images/epic.png" width="500"> <img src="./images/oln_overview.png" width="500"> <br/>

## Cross-category generalization on COCO

We train OLN on COCO VOC categories, and test on non-VOC categories. Note our AR@k evaluation does not count those proposals on the 'seen' classes into the budget (k), to avoid evaluating recall on see-class objects.

|     Method     |  AUC  | AR@10 | AR@30 | AR@100 | AR@300 | AR@1000 | Download |
|:--------------:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|
|    OLN-Box     | 24.8  | 18.0  | 26.4  |  33.4  |  39.0  |  45.0   | [model](https://drive.google.com/uc?id=1uL6TRhpSILvWeR6DZ0x9K9VywrQXQvq9) |


## Disclaimer

This repo is tested under Python 3.7, PyTorch 1.7.0, Cuda 11.0, and mmcv==1.2.5.

## Installation

This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection).

You can use following commands to create conda env with related dependencies.
```
conda create -n oln python=3.7 -y
conda activate oln
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch -y
pip install mmcv-full
pip install -r requirements.txt
pip install -v -e .
```
Please also refer to [get_started.md](docs/get_started.md) for more details of installation.


## Prepare datasets

COCO dataset is available from official websites. It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to $OLN/data as below.
```
object_localization_network
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017

```


## Testing
Our trained models are available for download [here](https://drive.google.com/uc?id=1uL6TRhpSILvWeR6DZ0x9K9VywrQXQvq9). Place it under `trained_weights/latest.pth` and run the following commands to test OLN on COCO dataset.

```
# Multi-GPU distributed testing
bash tools/dist_test_bbox.sh configs/oln_box/oln_box.py \
trained_weights/latest.pth ${NUM_GPUS}
# OR
python tools/test.py configs/oln_box/oln_box.py work_dirs/oln_box/latest.pth --eval bbox
```

python tools/test.py configs/oln_mask/class_agn_mask_rcnn_pp.py /checkpoint/weiyaowang/pairwise_potential/noncoco_to_coco/48571008_iou0.5_top1/latest.pth --eval segm


## Training
```
# Multi-GPU distributed training
bash tools/dist_train.sh configs/oln_box/oln_box.py ${NUM_GPUS}

```

## PA Training
```
bash tools/dist_train.sh configs/pairwise_affinity/pa_train.py 2 --work-dir /checkpoint/weiyaowang/pairwise_potential/pa_test/pa_test_local
```

## PA Eval
python tools/test_pa.py configs/pairwise_affinity/pa_train.py /checkpoint/weiyaowang/pairwise_potential/pa_vanilla_param/pa_0.01lr_8ep_6sche/latest.pth --eval pa

## PA Extract masks
python tools/extract_pa_masks_no_ann.py configs/pairwise_affinity/pa_extract_openImages.py /checkpoint/weiyaowang/pairwise_potential/pa_upernet_augmentation/ba1_orig_scale/latest.pth --out /checkpoint/weiyaowang/pairwise_potential/pa_upernet_augmentation/ba1_orig_scale/test_masks.json


## Pseudo Mask Training
```
bash tools/dist_train.sh configs/oln_mask/class_agn_mask_rcnn_pp.py 2
```

## Two Tower Training
bash tools/dist_train.sh configs/oln_mask/two_tower.py 2 --work-dir /checkpoint/weiyaowang/pairwise_potential/two_tower_test/

## eval mask rcnn
python tools/test.py configs/oln_mask/two_tower.py /checkpoint/weiyaowang/pairwise_potential/two_tower_explore/46715159_FFFT/latest.pth --eval segm

## eval faster rcnn
python tools/test.py configs/oln_box/class_agn_faster_rcnn.py /checkpoint/weiyaowang/pairwise_potential/multinode_test/fasterrcnn/latest.pth --eval bbox

## PLOT TRAINING CURVE
python tools/analyze_logs.py plot_curve /checkpoint/weiyaowang/pairwise_potential/imagenet/49065366pp_top1_maskrcnn_gn_scratch_4xScale_lr0.02_ep72,64,70/20220301_015216.log.json --keys loss --out loss.pdf
