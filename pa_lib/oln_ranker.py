# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import json
import pickle as pkl
from typing import List, Tuple

import numpy as np
import torch
from pycocotools import mask as maskUtils

from .affinity2mask import potential2masks_ucm_hierarchy
from .evaluate_helper import (
    compute_group_size,
    compute_ranking_score,
    get_nms_remove_ind,
    filter_mask_by_edge,
)


def compute_rpn_score_oln(feat, proposals, oln_model, img_metas):
    cls_scores, bbox_preds, objectness_pred = oln_model.rpn_head(feat)
    featmap_sizes = [featmap.size()[-2:] for featmap in objectness_pred]
    anchor_list, valid_flag_list = oln_model.rpn_head.get_anchors(
        featmap_sizes,
        img_metas,
        device=f"cuda:{torch.cuda.current_device()}",
    )
    label_channels = 1
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    concat_anchor_list = []
    concat_valid_flag_list = []
    num_imgs = 1
    for i in range(num_imgs):
        concat_anchor_list.append(torch.cat(anchor_list[i]))
        concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

    gt_bboxes = (
        torch.from_numpy(proposals[:, :-1]).float().cuda(torch.cuda.current_device())
    )
    overlaps = oln_model.rpn_head.objectness_assigner.iou_calculator(
        gt_bboxes.cpu(), concat_anchor_list[0].cpu()
    )
    max_anchor = torch.argmax(overlaps, dim=1)
    flattened_score = torch.cat(
        [objectness.reshape(-1) for objectness in objectness_pred]
    )
    centerness_scores = flattened_score[max_anchor]
    centerness_scores = torch.sigmoid(centerness_scores)
    return centerness_scores


def compute_roi_score_oln(feat, img_metas, oln_model, proposals):
    pred = oln_model.roi_head.simple_test(
        feat,
        [torch.from_numpy(proposals).float().cuda(torch.cuda.current_device())],
        img_metas,
    )
    proposal_scores = pred[0][1][:, 0].detach().cpu().numpy()
    return proposal_scores


def generate_pa_proposals_with_oln(
    pred_pa,
    img,
    img_metas,
    oln_model,
    edge_thresh=0.1,
    join_thresh=0.9,
    rank_method=3,
    min_component_size=100,
    nms=0.5,
    filter_by_edge_method=2,
    filter_by_edge_thresh=0.2,
    use_orientation=0,
    use_globalization=0.0,
    use_new_affinity=False,
    **kwargs,
):
    pred_pa = pred_pa.cpu().detach().numpy()
    torch.cuda.empty_cache()
    max_affinity = np.min(pred_pa, axis=0)
    pred_palette, all_segments, new_affinity = potential2masks_ucm_hierarchy(
        max_affinity,
        local_max_threshold=edge_thresh,
        merge_threshold=join_thresh,
        use_orientation=use_orientation,
        use_globalization=use_globalization,
    )
    if use_new_affinity:
        max_affinity = new_affinity
    if pred_palette is None:
        return None, None
    run_nms = True
    if all_segments is None:
        all_segments = [[idx] for idx in np.unique(pred_palette)]
        run_nms = False
    avg_map = {}
    mask_map = {}
    cnt = 0
    for i, segments in enumerate(all_segments):
        segment_mask = np.isin(pred_palette, list(segments))
        if np.sum(segment_mask) < min_component_size:
            continue
        if filter_by_edge_method and filter_mask_by_edge(
            segment_mask, method=filter_by_edge_method, thresh=filter_by_edge_thresh
        ):
            continue
        mask_map[i] = segment_mask
        avg_map[i] = compute_ranking_score(segment_mask, pred_pa, rank_method)
    pa_masks = list(mask_map.values())
    pa_scores = 1 - np.array(list(avg_map.values()))
    device = next(oln_model.parameters()).device
    feat = oln_model.extract_feat(img.data[0].cuda(device))

    pa_masks = np.stack(pa_masks)[:, :, :]
    masks = np.transpose(pa_masks[:, :, :], (1, 2, 0))
    rles = maskUtils.encode(np.asfortranarray(masks))
    areas = maskUtils.area(rles)
    bboxes = maskUtils.toBbox(rles)

    proposals = np.zeros_like(bboxes)
    proposals[:, 0] = bboxes[:, 0]
    proposals[:, 1] = bboxes[:, 1]
    proposals[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    proposals[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    scores = np.ones_like(proposals[:, 0:1])
    proposals = np.concatenate([proposals, scores], axis=1)

    proposal_scores = compute_roi_score_oln(
        feat, img_metas.data[0], oln_model, proposals
    )
    centerness_scores = compute_rpn_score_oln(
        feat, proposals, oln_model, img_metas.data[0]
    )
    centerness_scores = centerness_scores.detach().cpu().numpy()
    combined_scores = (
        pa_scores / 2 + proposal_scores + proposal_scores * centerness_scores
    )
    #     combined_scores = pa_scores / 2 * proposal_scores * proposal_scores
    for i, obj_ind in enumerate(list(avg_map.keys())):
        avg_map[obj_ind] = -combined_scores[i]
    avg_map = dict(sorted(avg_map.items(), key=lambda item: item[1]))
    if run_nms and nms > 0:
        size_map = compute_group_size(pred_palette)
        obj_indices = list(avg_map.keys())
        remove_obj_ind = get_nms_remove_ind(size_map, obj_indices, all_segments, nms)
        for obj_ind in remove_obj_ind:
            del avg_map[obj_ind], mask_map[obj_ind]
    masks = []
    scores = []
    for label_idx, avg_pp in avg_map.items():
        masks.append(mask_map[label_idx])
        scores.append(-avg_pp)
    if len(masks) == 0:
        print(f"empty masks: {np.unique(pred_palette)}")
        return None, None
    return masks, scores
