# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from .accuracy import Accuracy, accuracy
from .ae_loss import AssociativeEmbeddingLoss
from .balanced_l1_loss import balanced_l1_loss, BalancedL1Loss
from .cross_entropy_loss import (
    binary_cross_entropy,
    cross_entropy,
    CrossEntropyLoss,
    mask_cross_entropy,
)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .gaussian_focal_loss import GaussianFocalLoss
from .gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from .ghm_loss import GHMC, GHMR
from .iou_loss import (
    bounded_iou_loss,
    BoundedIoULoss,
    CIoULoss,
    DIoULoss,
    GIoULoss,
    iou_loss,
    IoULoss,
)
from .mse_loss import mse_loss, MSELoss
from .pisa_loss import carl_loss, isr_p
from .smooth_l1_loss import l1_loss, L1Loss, smooth_l1_loss, SmoothL1Loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .varifocal_loss import VarifocalLoss

__all__ = [
    "accuracy",
    "Accuracy",
    "cross_entropy",
    "binary_cross_entropy",
    "mask_cross_entropy",
    "CrossEntropyLoss",
    "sigmoid_focal_loss",
    "FocalLoss",
    "smooth_l1_loss",
    "SmoothL1Loss",
    "balanced_l1_loss",
    "BalancedL1Loss",
    "mse_loss",
    "MSELoss",
    "iou_loss",
    "bounded_iou_loss",
    "IoULoss",
    "BoundedIoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
    "GHMC",
    "GHMR",
    "reduce_loss",
    "weight_reduce_loss",
    "weighted_loss",
    "L1Loss",
    "l1_loss",
    "isr_p",
    "carl_loss",
    "AssociativeEmbeddingLoss",
    "GaussianFocalLoss",
    "QualityFocalLoss",
    "DistributionFocalLoss",
    "VarifocalLoss",
]
