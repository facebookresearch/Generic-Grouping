# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import (
    DETECTORS,
    HEADS,
    build_backbone,
    build_head,
    build_neck,
    build_loss,
)
from ..losses.focal_loss import FocalLoss
from .base import BaseDetector

EPS = 1e-5


def masks2palette(masks):
    """
    Convert binary object masks to palette
    Zero indicates background (no annotations)
    Args:
        masks: np.array, NxHxW, where N indicates number of objects
    """
    palette = np.zeros(masks.shape[1:], dtype=np.int8)
    for i, mask in enumerate(masks):
        palette += mask * (i + 1)
    return palette


def palette2filter(palette, neighbor_sizes=None, bidirection=True):
    """
    Generate indicator function to filter out background in loss.
    Same shape as PA target, with 1 as object-object/bg, 0 as bg-bg.
    Args:
        palette: segmentation palette
        neighbor_sizes: the distance used to compute pairwise relations
        bidirection: if True, generate across all (8) neighbors per distance.
            Otherwise, generate 4 (due to symmetry)
    """

    def generate_cross_filter(number_of_neighbors, dist):
        cross_filter = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.bool)
        cross_filter[0, dist:, :] = np.logical_or(palette[dist:, :], palette[:-dist, :])
        cross_filter[1, :, dist:] = np.logical_or(palette[:, dist:], palette[:, :-dist])
        cross_filter[2, dist:, dist:] = np.logical_or(
            palette[dist:, dist:], palette[:-dist, :-dist]
        )
        cross_filter[3, :-dist, dist:] = np.logical_or(
            palette[:-dist, dist:], palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_filter[4, :-dist, :] = np.logical_or(
                palette[:-dist, :], palette[dist:, :]
            )
            cross_filter[5, :, :-dist] = np.logical_or(
                palette[:, dist:], palette[:, :-dist]
            )
            cross_filter[6, :-dist, :-dist] = np.logical_or(
                palette[dist:, dist:], palette[:-dist, :-dist]
            )
            cross_filter[7, dist:, :-dist] = np.logical_or(
                palette[:-dist, dist:], palette[dist:, :-dist]
            )
        return cross_filter

    # manual assign default to avoid list input
    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    potential_filter = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.bool
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        potential_filter[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_filter(number_of_neighbors_per_span, dist)
    return potential_filter


def palette2weight(
    palette, neighbor_sizes=None, bidirection=True, weight_type="uniform"
):
    """
    Use to weight per-pixel loss of PA training (NOT used in paper)
    Weight Type supports:
        1. Uniform weight
        2. Inverse weight
        3. Square root inverse weight (inverse_sqrt)
    """

    def generate_cross_weight(number_of_neighbors, dist):
        cross_filter = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.bool)
        cross_filter[0, dist:, :] = np.maximum(palette[dist:, :], palette[:-dist, :])
        cross_filter[1, :, dist:] = np.maximum(palette[:, dist:], palette[:, :-dist])
        cross_filter[2, dist:, dist:] = np.maximum(
            palette[dist:, dist:], palette[:-dist, :-dist]
        )
        cross_filter[3, :-dist, dist:] = np.maximum(
            palette[:-dist, dist:], palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_filter[4, :-dist, :] = np.maximum(
                palette[:-dist, :], palette[dist:, :]
            )
            cross_filter[5, :, :-dist] = np.maximum(
                palette[:, dist:], palette[:, :-dist]
            )
            cross_filter[6, :-dist, :-dist] = np.maximum(
                palette[dist:, dist:], palette[:-dist, :-dist]
            )
            cross_filter[7, dist:, :-dist] = np.maximum(
                palette[:-dist, dist:], palette[dist:, :-dist]
            )
        return cross_filter

    weight = np.ones_like(palette)
    if weight_type != "uniform":
        mask_inds = np.unique(palette)
        for mask_idx in mask_inds:
            binary_mask = palette == mask_idx
            if mask_idx == 0:
                weight[binary_mask] = 0.0
                continue
            mask_cnt = np.sum(palette == mask_idx)
            mask_weight = mask_cnt
            if weight_type == "inverse":
                mask_weight /= 1.0
            elif weight_type == "inverse_sqrt":
                mask_weight = np.sqrt(mask_weight)
                mask_weight /= 1.0
            else:
                print("unsupported loss weight type, use uniform")
                mask_weight = 1.0
            weight[binary_mask] = mask_weight

    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    potential_weight = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.bool
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        potential_weight[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_weight(number_of_neighbors_per_span, dist)
    return potential_weight


def palette2affinity(palette, neighbor_sizes=None, bidirection=True):
    """
    convert palette (HxW, np) to pairwise affinity
    """

    def generate_cross_affinity(number_of_neighbors, dist):
        """
        generate single scale affinity
        """
        cross_affinity = np.zeros((number_of_neighbors,) + palette.shape, dtype=np.int8)
        # top
        cross_affinity[0, dist:, :] = palette[dist:, :] == palette[:-dist, :]
        # left
        cross_affinity[1, :, dist:] = palette[:, dist:] == palette[:, :-dist]
        # left, top
        cross_affinity[2, dist:, dist:] = (
            palette[dist:, dist:] == palette[:-dist, :-dist]
        )
        # left, down
        cross_affinity[3, :-dist, dist:] = (
            palette[:-dist, dist:] == palette[dist:, :-dist]
        )
        if number_of_neighbors == 8:
            cross_affinity[4, :-dist, :] = palette[dist:, :] == palette[:-dist, :]
            cross_affinity[5, :, :-dist] = palette[:, dist:] == palette[:, :-dist]
            cross_affinity[6, :-dist, :-dist] = (
                palette[dist:, dist:] == palette[:-dist, :-dist]
            )
            cross_affinity[7, dist:, :-dist] = (
                palette[:-dist, dist:] == palette[dist:, :-dist]
            )
        return cross_affinity

    # manual assign default to avoid list input
    if neighbor_sizes is None:
        neighbor_sizes = [1]
    number_of_span = len(neighbor_sizes)
    number_of_neighbors_per_span = 8 if bidirection else 4
    pairwise_affinity = np.zeros(
        (number_of_neighbors_per_span * number_of_span,) + palette.shape, dtype=np.int8
    )
    for neighbor_idx, dist in enumerate(neighbor_sizes):
        offset = neighbor_idx * number_of_neighbors_per_span
        pairwise_affinity[
            offset : offset + number_of_neighbors_per_span
        ] = generate_cross_affinity(number_of_neighbors_per_span, dist)
    return pairwise_affinity


def collapse_affinity(affinity, bg_filter, reduction=None):
    """
    Pooling used to convert affinity into a 1-channel boundary probability
    """
    if reduction is None:
        return affinity, bg_filter
    bg_filter = np.max(bg_filter, axis=0, keepdims=True)
    if reduction == "mean":
        affinity = np.mean(affinity.astype(float), axis=0, keepdims=True)
    elif reduction == "min":
        affinity = np.min(affinity, axis=0, keepdims=True)
    elif reduction == "max":
        affinity = np.max(affinity, axis=0, keepdims=True)
    return affinity, bg_filter


def instance_mask2affinity(
    instance_mask, bidirection=True, reduction=None, weight_type="uniform"
):
    masks = instance_mask.to_ndarray()
    palette = masks2palette(masks)
    bg_filter = palette2weight(
        palette, bidirection=bidirection, weight_type=weight_type
    )
    affinity = palette2affinity(palette, bidirection=bidirection)
    affinity, bg_filter = collapse_affinity(affinity, bg_filter, reduction=reduction)
    return torch.from_numpy(affinity), torch.from_numpy(bg_filter)


@HEADS.register_module()
class PairwiseAffinityHead(nn.Sequential):
    """
    Pairwise affinity predictor
    """

    def __init__(
        self,
        in_channels=2048,
        channels=8,
        norm_type="BN",
    ):
        inter_channels = in_channels // 4
        if norm_type == "BN":
            norm_layer = nn.BatchNorm2d(inter_channels)
        else:
            norm_layer = nn.GroupNorm(32, inter_channels)
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super(PairwiseAffinityHead, self).__init__(*layers)


@HEADS.register_module()
class PairwiseAffinityHeadUperNet(nn.Module):
    """
    UperNet version of predictor head; leverage FPN
    Reference: https://github.com/CSAILVision/unifiedparsing
    """

    def __init__(
        self,
        channels=8,
        fc_dim=2048,
        pool_scales=(1, 2, 3, 6),
        fpn_inplanes=(256, 512, 1024, 2048),
        fpn_dim=256,
        norm_type="BN",
    ):
        super(PairwiseAffinityHeadUperNet, self).__init__()

        # helper block
        def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm_layer="BN"):
            "3x3 convolution + BN + relu"
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(out_planes)
            else:
                norm_layer = nn.GroupNorm(32, out_planes)
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                norm_layer,
                nn.ReLU(inplace=True),
            )

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(512)
            else:
                norm_layer = nn.GroupNorm(32, 512)
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(
            fc_dim + len(pool_scales) * 512, fpn_dim, 1
        )

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            if norm_type == "BN":
                norm_layer = nn.BatchNorm2d(fpn_dim)
            else:
                norm_layer = nn.GroupNorm(32, fpn_dim)
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, channels, kernel_size=1),
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    nn.functional.interpolate(
                        pool_scale(conv5),
                        (input_size[2], input_size[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode="bilinear", align_corners=False
            )  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(
                nn.functional.interpolate(
                    fpn_feature_list[i],
                    output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        return x


@DETECTORS.register_module()
class PairwiseAffinityPredictor(BaseDetector):
    """Base class for pairwise affinity predictor.

    Pairwise affinity predictor contains a single feature extractor with
        PA predicting head
    """

    def __init__(
        self,
        backbone,
        neck=None,
        classifier=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        loss_affinity=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            class_weight=[0.05],
            reduction="none",
        ),
        bidirection=True,
        only_last_conv_feat=False,
        affinity_reduction=None,
        filter_bg=True,
        weight_type="uniform",
        norm_type="BN",
    ):
        """
        Args:
            classifier: Head to map features into PA predictions
            loss_affinity: loss used to train PA;
                paper used CrossEntropyLoss with positive weight 0.05
                experimented with FocalLoss, but gain is limited
            only_last_conv_feat: if only return last conv stage of backbone
                True if using PairwiseAffinityHead
                False if using PairwiseAffinityHeadUperNet
            filter_bg: if applying bg-bg filtering
        """
        super(PairwiseAffinityPredictor, self).__init__()
        self.norm_type = norm_type
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if classifier is not None:
            self.classifier = build_head(classifier)
        else:
            self.classifier = None

        # TODO: remove if not used
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)
        # should be no_reduction
        self.loss_affinity = build_loss(loss_affinity)
        self.bidirection = bidirection
        self.only_last_conv_feat = only_last_conv_feat
        self.affinity_reduction = affinity_reduction
        self.filter_bg = filter_bg
        self.weight_type = weight_type

    # override since PA always requires mask
    @property
    def with_mask(self):
        return True

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(PairwiseAffinityPredictor, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        # NOTE: no init_weight needed for classifier for now
        # if self.classifier:
        #     self.classifier.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if self.only_last_conv_feat:
            return x[-1]
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        return x

    # TODO: optimize this step as GPU ops
    def preprocess_gt(self, gt_masks, device):
        affinity_filter_maps = [
            instance_mask2affinity(
                inst_mask, self.bidirection, self.affinity_reduction, self.weight_type
            )
            for inst_mask in gt_masks
        ]
        affinity_maps = [
            affinity_filter[0].to(device) for affinity_filter in affinity_filter_maps
        ]
        filter_maps = [
            affinity_filter[1].to(device) for affinity_filter in affinity_filter_maps
        ]
        return affinity_maps, filter_maps

    def forward_train(self, img, img_metas, gt_masks, **kwargs):
        x = self.extract_feat(img)
        device = x[0].device
        losses = dict()
        affinity_maps, filter_maps = self.preprocess_gt(gt_masks, device)
        pred_pa = self.classifier(x)
        input_shape = img.shape[-2:]
        pred_pa = F.interpolate(
            pred_pa, size=input_shape, mode="bilinear", align_corners=False
        )
        affinity_losses = self.compute_loss(pred_pa, affinity_maps, filter_maps)
        losses.update(affinity_losses)
        return losses

    # Override forward test
    def forward_test(self, img, img_metas, gt_masks=None, **kwargs):
        x = self.extract_feat(img)
        device = x[0].device
        if gt_masks is not None:
            affinity_maps, filter_maps = self.preprocess_gt(gt_masks, device)
        pred_pa = self.classifier(x)
        input_shape = img.shape[-2:]
        pred_pa = F.interpolate(
            pred_pa, size=input_shape, mode="bilinear", align_corners=False
        )
        if gt_masks is None:
            pred_pa = F.sigmoid(pred_pa)
            return None, pred_pa, None
        affinity_quality, pa_preds = self.compute_pa_quality(
            pred_pa, affinity_maps, filter_maps
        )
        return affinity_quality, pa_preds, gt_masks

    def compute_loss(self, pred_pa, gt_affinity, bg_filter):
        loss = dict()
        loss_pa = 0.0
        for pred, gt, bg_f in zip(pred_pa, gt_affinity, bg_filter):
            # since batching > 1 image may involve padding, remove the pad
            h, w = gt.shape[-2], gt.shape[-1]
            if type(self.loss_affinity) is FocalLoss:
                per_loc_loss = self.loss_affinity(
                    torch.flatten(pred[:, :h, :w]).reshape(-1, 1),
                    torch.flatten(gt.long()),
                )
                per_loc_loss = torch.reshape(per_loc_loss, (-1, h, w))
            else:
                per_loc_loss = self.loss_affinity(pred[:, :h, :w], gt)
            if self.filter_bg:
                total_loss = torch.sum(per_loc_loss * bg_f)
                normalizer = torch.sum(bg_f) + EPS
                loss_pa += total_loss / normalizer
            else:
                loss_pa += torch.mean(per_loc_loss)
        loss["loss_pa"] = loss_pa / pred_pa.shape[0]
        return loss

    def compute_pa_quality(self, pred_pa, gt_affinity, bg_filter):
        """
        Compute PA quality (CE/L1 loss) to evaluate PA independent of grouping
        NOT used in paper
        """
        pa_qualities = []
        pa_preds = []
        for pred, gt, bg_f in zip(pred_pa, gt_affinity, bg_filter):
            # since batching > 1 image may involve padding, remove the pad
            pa_quality = dict()
            h, w = gt.shape[-2], gt.shape[-1]
            concat_pred = pred[:, :h, :w]
            pa_ce = F.binary_cross_entropy_with_logits(
                concat_pred, gt.float(), reduction="none"
            )
            sigmoid_pa = F.sigmoid(concat_pred)
            pa_l1 = F.l1_loss(sigmoid_pa, gt.float(), reduction="none")
            inner_filter = (gt == 1).float() * bg_f
            outer_filter = (gt == 0).float() * bg_f
            inner_norm = torch.sum(inner_filter) + EPS
            outer_norm = torch.sum(outer_filter) + EPS
            inner_pa_ce = torch.sum(pa_ce * inner_filter) / inner_norm
            outer_pa_ce = torch.sum(pa_ce * outer_filter) / outer_norm
            inner_pa_l1 = torch.sum(pa_l1 * inner_filter) / inner_norm
            outer_pa_l1 = torch.sum(pa_l1 * outer_filter) / outer_norm
            pa_quality["inner_pa_ce"] = inner_pa_ce
            pa_quality["outer_pa_ce"] = outer_pa_ce
            pa_quality["inner_pa_l1"] = inner_pa_l1
            pa_quality["outer_pa_l1"] = outer_pa_l1
            pa_qualities.append(pa_quality)
            pa_preds.append(sigmoid_pa)

        return pa_qualities, pa_preds

    # TODO: implement inference and test
    def simple_test(self, imgs, img_metas):
        return None

    def aug_test(self, imgs, img_metas):
        return None
