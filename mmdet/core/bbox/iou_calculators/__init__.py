# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from .builder import build_iou_calculator
from .iou2d_calculator import bbox_overlaps, BboxOverlaps2D

__all__ = ["build_iou_calculator", "BboxOverlaps2D", "bbox_overlaps"]
