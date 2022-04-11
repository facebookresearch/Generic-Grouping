# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
    filter_bg=True,
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
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_labels", "gt_bboxes", "gt_masks"],
        meta_keys=[
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Normalize", **img_norm_cfg),
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
            "pad_shape",
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class="voc",
        eval_class="nonvoc",
        ann_file=data_root + "annotations/instances_train2017.json",
        img_prefix=data_root + "train2017/",
        type=dataset_type,
        pipeline=train_pipeline,
    ),
    val=dict(
        is_class_agnostic=True,
        train_class="voc",
        eval_class="nonvoc",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        type=dataset_type,
        pipeline=test_pipeline,
    ),
    test=dict(
        is_class_agnostic=True,
        train_class="nonvoc",
        eval_class="voc",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        type=dataset_type,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=1, metric=["bbox", "segm"])

optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

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
gpu_ids = range(0, 8)

# Use this if you are evaluating PA and need to set parameters
# pa_to_masks = dict(
#     edge_thresh=0.0,
#     join_thresh=1.0,
#     rank_method=3,
#     nms=0.2,
#     use_orientation=1,
#     use_globalization=0.5,
#     use_new_affinity=False,
#     filter_by_edge_method=2,
#     filter_by_edge_thresh=0.9,
#     min_component_size=0,
#     oln_ranker_path=None,
# )
