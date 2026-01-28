# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.models import build_detector

cfg_file = "./configs/oln_mask/two_tower_example.py"
checkpoint = "/checkpoint/weiyaowang/pairwise_potential/coco_to_lvis/maskrcnn_baseline/latest.pth"
img_path = "/checkpoint/trandu/oln/data/coco/train2017/000000391895.jpg"

model_config = Config.fromfile(cfg_file).model
model_config.test_cfg.rcnn.nms = dict(type="nms", iou_threshold=1.0)
two_tower = build_detector(model_config)
# load_checkpoint(mask_rcnn, checkpoint, map_location='cpu')
two_tower.cpu()
two_tower.eval()
input_img = cv2.imread(img_path)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
input_img = mmcv.imnormalize(
    input_img,
    np.array(img_norm_cfg["mean"]),
    np.array(img_norm_cfg["std"]),
)
img_metas = [
    {
        "img_shape": input_img.shape,
        "scale_factor": 1.0,
        "ori_shape": input_img.shape,
        "pad_shape": input_img.shape,
    }
]

input_img = np.transpose(input_img, (2, 0, 1))
input_img = torch.from_numpy(input_img)

input_img = input_img.unsqueeze(0)

out = two_tower.simple_test(input_img, img_metas)

# features = mask_rcnn.extract_feat(input_img)

# # (tl_x, tl_y, br_x, br_y)
# proposal_list = mask_rcnn.rpn_head.simple_test_rpn(features, img_metas)
# det_bboxes, det_labels = mask_rcnn.roi_head.simple_test_bboxes(
#     features, img_metas, proposal_list, mask_rcnn.roi_head.test_cfg, rescale=False
# )
# segm_results = mask_rcnn.roi_head.simple_test_mask(
#     features, img_metas, det_bboxes, det_labels, rescale=False
# )
# print(proposal_list[0].shape)
# roi_out = mask_rcnn.roi_head.forward_dummy(features, proposal_list[0])
# print(roi_out[0].shape)
# print(roi_out[1].shape)
# print(roi_out[2].shape)
# print(roi_out[3].shape)

# print(f"det boxes: {det_bboxes[0].shape}")

# print(proposal_list[0][::200, :4])
# print(det_bboxes[0][::200, :4])
# print(segm_results[0].shape)

# print(roi_out[1][::200, :4])
# print(roi_out[2][0, 0, :, :])
# print(roi_out[1][:5, :4])
# print(roi_out[3][:5, :4])
