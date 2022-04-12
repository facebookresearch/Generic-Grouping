# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from mmcv.utils import Registry, build_from_cfg

IOU_CALCULATORS = Registry("IoU calculator")


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
