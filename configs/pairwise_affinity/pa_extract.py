# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

_base_ = [
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
# model settings
model = dict(
    type="PairwiseAffinityPredictor",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        # norm_cfg=dict(type="GN", requires_grad=True, num_groups=32),
        norm_eval=False,
        style="pytorch",
        strides=(1, 2, 2, 2),
    ),
    classifier=dict(
        type="PairwiseAffinityHeadUperNet",
        channels=1,
        # norm_type='GN',
    ),
    loss_affinity=dict(
        type="CrossEntropyLoss",
        use_sigmoid=True,
        class_weight=[0.05],
        reduction="none",
    ),
    affinity_reduction="mean",
)

# Dataset
dataset_type = "CocoSplitDataset"
data_root = "data/coco/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Resize", img_scale=(1333, 800), keep_ratio=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "scale_factor",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    # comment below to use ImageNet
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Normalize", **img_norm_cfg),
    # uncomment if want ImageNet scale
    # dict(type="Resize", img_scale=(640, 400), keep_ratio=True),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks"],
        # comment above and uncomment below for ImageNet (no GT anns)
        # keys=["img"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "scale_factor",
            "pad_shape",
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        is_class_agnostic=True,
        train_class="voc",
        eval_class="voc",
        ann_file=data_root + "annotations/instances_train2017.json",
        img_prefix=data_root + "train2017/",
        type=dataset_type,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric=["bbox", "segm"])

optimizer = dict(type="SGD", lr=0.04, momentum=0.9, weight_decay=0.0001)

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[15]
)
total_epochs = 20

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# parameters used to extract PA masks
pa_to_masks = dict(
    edge_thresh=0.0,
    join_thresh=1.0,
    rank_method=3,
    nms=0.2,
    use_orientation=1,
    use_globalization=0.5,
    use_new_affinity=False,
    filter_by_edge_method=2,
    filter_by_edge_thresh=0.9,
    min_component_size=0,
    # Set to None if don't want to use OLN ranker
    oln_ranker_path=None,
)
