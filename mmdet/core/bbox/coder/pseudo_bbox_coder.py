# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        super(BaseBBoxCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        """torch.Tensor: return the given ``bboxes``"""
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        """torch.Tensor: return the given ``pred_bboxes``"""
        return pred_bboxes
