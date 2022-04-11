_base_ = "coco_detection.py"
dataset_type = "LVISV05Dataset"
data_root = "data/LVIS/"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type="ClassBalancedDataset",
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            # ann_file=data_root + 'annotations/lvis_v0.5_train.json',
            ann_file=data_root + "annotations/lvis_v0.5_train_10.json",
            img_prefix=data_root + "train2017/",
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/lvis_v0.5_val.json",
        img_prefix=data_root + "val2017/",
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/lvis_v0.5_val.json",
        img_prefix=data_root + "val2017/",
    ),
)
evaluation = dict(metric=["bbox", "segm"])
