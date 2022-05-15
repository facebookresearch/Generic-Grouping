# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..builder import build_backbone, build_detector, build_head, build_neck, DETECTORS
from .base import BaseDetector

RPN_SCORE_LAYERS = [
    "rpn_head.rpn_cls.weight",
    "rpn_head.rpn_cls.bias",
    "rpn_head.rpn_obj.weight",
    "rpn_head.rpn_obj.bias",
]


# NOTE: Batchsize = 1
@DETECTORS.register_module()
class TwoTowerDetector(BaseDetector):
    def __init__(
        self,
        rec_backbone,
        group_detector,
        rec_neck=None,
        rec_roi_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        refine_grouping=False,
        pretrained_grouping=None,
        freeze_rpn_scoring=False,
        feed_in_rpn=False,
        freeze_backbone=False,
        rpn_test_mode=False,
    ):
        super(TwoTowerDetector, self).__init__()
        if rpn_test_mode:
            freeze_rpn_scoring = True
        self.rpn_test_mode = rpn_test_mode
        self.rec_backbone = build_backbone(rec_backbone)
        self.group_detector = build_detector(group_detector)
        if rec_neck is not None:
            self.neck = build_neck(rec_neck)
        if rec_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            rec_roi_head.update(train_cfg=rcnn_train_cfg)
            rec_roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(rec_roi_head)

        if not refine_grouping:
            mask_head = self.roi_head.mask_head
            del mask_head
            self.roi_head.mask_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.g_cfg = group_detector
        self.refine_grouping = refine_grouping
        self.init_weights(
            pretrained=pretrained,
            pretrained_grouping=pretrained_grouping,
            freeze_backbone=freeze_backbone,
            freeze_rpn_scoring=freeze_rpn_scoring,
        )
        self.freeze_backbone = freeze_backbone
        self.debug = None
        self.feed_in_rpn = feed_in_rpn

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return False

    @property
    def with_roi_head(self):
        return hasattr(self, "roi_head") and self.roi_head is not None

    def init_weights(
        self,
        pretrained=None,
        pretrained_grouping=None,
        freeze_backbone=False,
        freeze_rpn_scoring=False,
    ):
        # TODO: set up different way to initialize Group tower
        # NOTE: check if trainer will manually set the model.train()
        super(TwoTowerDetector, self).init_weights(pretrained)
        self.rec_backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)
        # TODO: better way to load weight
        load_checkpoint(self.group_detector, pretrained_grouping, map_location="cpu")
        for name, param in self.group_detector.named_parameters():
            if freeze_rpn_scoring or name not in RPN_SCORE_LAYERS:
                param.requires_grad = False
        if freeze_backbone:
            for name, param in self.rec_backbone.named_parameters():
                param.requires_grad = False

    def extract_feat(self, img):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.rec_backbone(img)
        else:
            x = self.rec_backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_grouping(
        self,
        img,
        img_metas,
        gt_bboxes=None,
        gt_labels=None,
        gt_bboxes_ignore=None,
        training=False,
        rescale=False,
    ):
        # TODO: check for methods to use GT for RPN choosing proposals
        with torch.no_grad():
            g_feat = self.group_detector.extract_feat(img)
            proposal_cfg = self.g_cfg.train_cfg.get(
                "rpn_proposal", self.g_cfg.test_cfg.rpn
            )
        if not training or self.rpn_test_mode:
            with torch.no_grad():
                proposal_list = self.group_detector.rpn_head.simple_test_rpn(
                    g_feat, img_metas
                )
                rpn_losses = None
        else:
            rpn_losses, proposal_list = self.group_detector.rpn_head.forward_train(
                g_feat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
        if self.feed_in_rpn:
            return proposal_list, None, rpn_losses, None
        with torch.no_grad():
            det_bboxes, det_labels = self.group_detector.roi_head.simple_test_bboxes(
                g_feat,
                img_metas,
                proposal_list,
                self.group_detector.roi_head.test_cfg,
                rescale=rescale,
            )

            seg_result, mask_preds = self.group_detector.roi_head.simple_test_mask(
                g_feat,
                img_metas,
                det_bboxes,
                det_labels,
                return_raw_pred=True,
                rescale=rescale,
            )

        # TODO: Important; handle re-scale
        return det_bboxes, mask_preds, rpn_losses, seg_result

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        proposals=None,
        **kwargs,
    ):
        losses = dict()
        proposal_list, mask_list, rpn_losses, _ = self.extract_grouping(
            img,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            training=True,
            rescale=False,
        )

        if rpn_losses is not None:
            for loss in rpn_losses:
                losses["grouping_" + loss] = rpn_losses[loss]

        if mask_list is not None:
            mask_list = mask_list[0]

        x = self.extract_feat(img)
        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            mask_list,
        )
        losses.update(roi_losses)

        return losses

    def simple_test(
        self,
        img,
        img_metas,
        proposals=None,
        rescale=False,
    ):
        proposal_list, mask_list, _, seg_result = self.extract_grouping(img, img_metas)

        if self.refine_grouping:
            del seg_result

        if mask_list is not None:
            mask_list = mask_list[0]

        x = self.extract_feat(img)
        if self.refine_grouping:
            return self.roi_head.simple_test(
                x,
                proposal_list,
                img_metas,
                rescale=rescale,
                masks_list=mask_list,
            )
        else:
            det_bboxes, det_labels, det_nms_keep = self.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposal_list,
                self.roi_head.test_cfg,
                rescale=rescale,
                masks_list=mask_list,
            )
            det_scores = [det_bbox[:, -1] for det_bbox in det_bboxes]
            return self.roi_head.merge_results(
                proposal_list,
                det_labels,
                seg_result,
                det_nms_keep,
                det_scores,
            )

    # TODO: Implement multi-aug test
    def aug_test(self, img, img_metas, proposals=None, rescale=False):
        return self.simple_test(img, img_metas, proposals, rescale)
