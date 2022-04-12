# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class RecRoIHead(StandardRoIHead):
    def __init__(self, use_group_mask=False, stop_grad_group=True, *args, **kwargs):
        super(RecRoIHead, self).__init__(*args, **kwargs)
        self.use_group_mask = use_group_mask
        self.stop_grad_group = stop_grad_group

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        masks_list=None,
    ):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
                )
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x],
                )
                sampling_results.append(sampling_result)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas,
                masks_list,
            )
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            feature = x
            mask_results = self._mask_forward_train(
                feature,
                sampling_results,
                bbox_results["bbox_feats"],
                gt_masks,
                img_metas,
                masks_list,
            )
            losses.update(mask_results["loss_mask"])

        return losses

    def _bbox_forward(self, x, rois, masks_list):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(
            x[: self.bbox_roi_extractor.num_inputs], rois
        )
        if self.use_group_mask:
            if masks_list is None:
                masks_list = torch.zeros_like(bbox_feats[:, 0:1, ...])
            else:
                masks_list = F.interpolate(
                    masks_list,
                    size=bbox_feats.shape[-2:],
                    mode="bilinear",
                )
            bbox_feats = torch.cat([bbox_feats, masks_list], dim=1)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats
        )
        return bbox_results

    def _bbox_forward_train(
        self,
        x,
        sampling_results,
        gt_bboxes,
        gt_labels,
        img_metas,
        masks_list,
    ):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if masks_list is not None and self.use_group_mask:
            masks_list = torch.cat(
                [
                    masks_list[sampling_results[0].pos_inds],
                    masks_list[sampling_results[0].neg_inds],
                ]
            )
        bbox_results = self._bbox_forward(x, rois, masks_list)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg
        )
        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward(
        self,
        x,
        rois=None,
        pos_inds=None,
        bbox_feats=None,
        masks_list=None,
    ):
        """Mask head forward function used in both training and testing."""
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois
            )
            if self.use_group_mask:
                if masks_list is None:
                    masks_list = torch.zeros_like(mask_feats[:, 0:1, ...])
                else:
                    masks_list = F.interpolate(
                        masks_list,
                        size=mask_feats.shape[-2:],
                        mode="bilinear",
                    )
                mask_feats = torch.cat([mask_feats, masks_list], dim=1)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        if self.stop_grad_group:
            mask_feats = mask_feats.detach()
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train(
        self,
        x,
        sampling_results,
        bbox_feats,
        gt_masks,
        img_metas,
        masks_list,
    ):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if self.use_group_mask and masks_list is not None:
                masks_list = masks_list[sampling_results[0].pos_inds]
            mask_results = self._mask_forward(x, pos_rois, masks_list=masks_list)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0], device=device, dtype=torch.uint8
                    )
                )
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0], device=device, dtype=torch.uint8
                    )
                )
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats, masks_list=masks_list
            )

        mask_targets = self.mask_head.get_targets(
            sampling_results, gt_masks, self.train_cfg
        )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(
            mask_results["mask_pred"], mask_targets, pos_labels
        )

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def simple_test_bboxes(
        self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, masks_list=None
    ):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, masks_list)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results["cls_score"]
        bbox_pred = bbox_results["bbox_pred"]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img
                )
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_nms_keep = []
        for i in range(len(proposals)):
            det_bbox, det_label, keep = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
                return_inds=True,
            )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_nms_keep.append(keep)
        return det_bboxes, det_labels, det_nms_keep

    def simple_test_mask(
        self,
        x,
        img_metas,
        det_bboxes,
        det_labels,
        rescale=False,
        return_raw_pred=False,
        masks_list=None,
    ):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta["ori_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)
        num_imgs = len(det_bboxes)
        mask_preds = None
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [
                [[] for _ in range(self.mask_head.num_classes)] for _ in range(num_imgs)
            ]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            if torch.onnx.is_in_onnx_export():
                # avoid mask_pred.split with static number of prediction
                mask_preds = []
                _bboxes = []
                for i, boxes in enumerate(det_bboxes):
                    boxes = boxes[:, :4]
                    if rescale:
                        boxes *= scale_factors[i]
                    _bboxes.append(boxes)
                    img_inds = boxes[:, :1].clone() * 0 + i
                    mask_rois = torch.cat([img_inds, boxes], dim=-1)
                    mask_result = self._mask_forward(
                        x, mask_rois, masks_list=masks_list
                    )
                    mask_preds.append(mask_result["mask_pred"])
            else:
                _bboxes = [
                    det_bboxes[i][:, :4] * scale_factors[i]
                    if rescale
                    else det_bboxes[i][:, :4]
                    for i in range(len(det_bboxes))
                ]
                mask_rois = bbox2roi(_bboxes)
                mask_results = self._mask_forward(x, mask_rois, masks_list=masks_list)
                mask_pred = mask_results["mask_pred"]
                # split batch mask prediction back to each image
                num_mask_roi_per_img = [det_bbox.shape[0] for det_bbox in det_bboxes]
                mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i],
                        _bboxes[i],
                        det_labels[i],
                        self.test_cfg,
                        ori_shapes[i],
                        scale_factors[i],
                        rescale,
                    )
                    segm_results.append(segm_result)
        if return_raw_pred:
            return segm_results, mask_preds
        return segm_results

    def simple_test(
        self,
        x,
        proposal_list,
        img_metas,
        proposals=None,
        rescale=False,
        masks_list=None,
    ):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        det_bboxes, det_labels, det_nms_keep = self.simple_test_bboxes(
            x,
            img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale,
            masks_list=masks_list,
        )
        if masks_list is not None:
            det_nms_keep = [
                (keep / self.bbox_head.num_classes).long() for keep in det_nms_keep
            ]
            masks_list = masks_list[det_nms_keep[0]]
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x,
                    img_metas,
                    det_bboxes,
                    det_labels,
                    rescale=rescale,
                    masks_list=masks_list,
                )
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                masks_list=masks_list,
            )
            return list(zip(bbox_results, segm_results))

    def merge_results(
        self,
        det_bboxes,
        det_labels,
        segm_results,
        det_nms_keep,
        det_scores,
    ):
        # return det_bboxes, det_labels, segm_results, det_nms_keep, det_scores
        # map indices back to original proposal index
        det_nms_keep = [
            (keep / self.bbox_head.num_classes).long() for keep in det_nms_keep
        ]
        det_bboxes = [det_bboxes[i][det_nms_keep[i]] for i in range(len(det_bboxes))]
        for i, det_bbox in enumerate(det_bboxes):
            det_bbox[:, -1] = det_scores[i]
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # assume segm_result is class-agnostic (binary)
        segm_results = [
            np.array(segm_result[0])[det_nms_keep[i].cpu().numpy()]
            for i, segm_result in enumerate(segm_results)
        ]
        for j, segm_result in enumerate(segm_results):
            cls_segm_result = [[] for _ in range(self.bbox_head.num_classes)]
            for i, segm in enumerate(segm_result):
                cls_segm_result[det_labels[j][i]].append(segm)
            segm_results[j] = cls_segm_result
        return list(zip(bbox_results, segm_results))
