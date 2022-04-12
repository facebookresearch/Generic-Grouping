# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import pickle as pkl
from typing import List, Tuple

import numpy as np
import skimage
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .affinity2mask import potential2masks_ucm_hierarchy

PROPOSALS = [1, 10, 30, 100, 300, 1000]
IOU_THRESH = np.linspace(
    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
)


def rank_by_variance(image, masks, weight_by_size=False, nms=1.0):
    # assume image is CHW
    variance = []
    for mask in masks:
        masked_pixels = image[:, mask]
        color_var = np.var(masked_pixels, axis=1)
        if weight_by_size:
            size = np.sum(mask)
            if size == 0:
                # to throw away
                variance.append(1000)
            variance.append(np.sum(color_var) / size)
        else:
            variance.append(np.sum(color_var))
    rank = np.argsort(variance)
    masks = [masks[i] for i in rank]
    to_remove = []
    if nms < 1:
        for i in range(len(masks) - 1):
            top_mask = masks[i]
            for j in range(i + 1, len(masks)):
                if j not in to_remove:
                    bot_mask = masks[j]
                    iou = compute_iou(top_mask, bot_mask)
                    if iou > nms:
                        to_remove.append(j)
    to_remove.sort(reverse=True)
    for idx in to_remove:
        del masks[idx]
    return masks


def compute_iou(annotation, segmentation, mask_threshold=0.0):
    if type(annotation) == torch.Tensor:
        annotation = annotation.numpy()
    annotation = annotation.astype(np.bool)
    segmentation = (segmentation > mask_threshold).astype(np.bool)

    if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / np.sum(
            (annotation | segmentation), dtype=np.float32
        )


def compute_ious(masks, pred_masks, mask_threshold=0.0):
    iou_matrix = np.zeros((len(pred_masks), len(masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, mask in enumerate(masks):
            mask_iou = compute_iou(mask, pred_mask, mask_threshold)
            iou_matrix[i, j] = mask_iou
    return iou_matrix


def greedy_match(gt_masks, pred_masks):
    matched_ious = []
    iou_matrix = compute_ious(gt_masks, pred_masks)
    for _ in range(len(gt_masks)):
        pred_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched_ious.append(iou_matrix[pred_idx, gt_idx])
        # remove matched results
        iou_matrix[:, gt_idx] *= 0
        iou_matrix[pred_idx, :] *= 0
    return matched_ious


def compute_recall_normal(masks, pred_masks, scores=None):
    iou_matrix = compute_ious(
        masks, pred_masks, mask_threshold=0.0 if scores is None else 0.5
    )
    if scores is None:
        ranked_ind = np.array(range(len(pred_masks)))
    else:
        ranked_ind = np.argsort(-scores)

    result = {}
    result["obj_cnt"] = len(masks)
    result["pred_cnt"] = len(pred_masks)
    for thresh in IOU_THRESH:
        mapped_gt = [False] * len(masks)
        obj_found = [False] * 1000
        for i, pred_idx in enumerate(ranked_ind[:1000]):
            mapped_idx = -1
            curr_iou = thresh
            for gt_idx in range(len(masks)):
                if iou_matrix[pred_idx, gt_idx] <= curr_iou or mapped_gt[gt_idx]:
                    continue
                curr_iou = iou_matrix[pred_idx, gt_idx]
                mapped_idx = gt_idx
            if mapped_idx == -1:
                continue
            mapped_gt[mapped_idx] = True
            obj_found[i] = 1
        obj_found = np.array(obj_found)
        cum_result = np.cumsum(obj_found)
        avg_obj_found = []
        for obj_limit in PROPOSALS:
            # avg_det = np.mean(cum_result[:obj_limit])
            max_det = cum_result[:obj_limit][-1]
            avg_obj_found.append(max_det)
        # Add greedy one
        avg_obj_found.append(cum_result[-1])
        result[thresh] = avg_obj_found
    return result


def compute_recall_greedy(masks, pred_masks):
    result = {}
    result["obj_cnt"] = len(masks)
    result["pred_cnt"] = len(pred_masks)
    matched_ious = greedy_match(masks, pred_masks)
    for thresh in IOU_THRESH:
        obj_found = matched_ious >= thresh
        cum_result = np.cumsum(obj_found)
        avg_obj_found = []
        for obj_limit in PROPOSALS:
            if cum_result.size == 0:
                avg_obj_found.append(0)
            else:
                max_det = cum_result[:obj_limit][-1]
                avg_obj_found.append(max_det)
        result[thresh] = avg_obj_found
    return result


def init_global_result(eval_method=1):
    result = {}
    if eval_method == 0:
        result["pp_loss_filtered"] = []
        result["pp_loss_raw"] = []
    elif eval_method == 3:
        result = []
    else:
        result["obj_cnt"] = []
        result["pred_cnt"] = []
        for thresh in IOU_THRESH:
            result[thresh] = []
    return result


def accumulate_result(results):
    agg_result = {}
    metrics = results[0].keys()
    for metric in metrics:
        if metric != "recall":
            all_metric_results = [result[metric].cpu().numpy() for result in results]
            agg_result[metric] = np.mean(all_metric_results)
        else:
            obj_cnt = np.sum([result["recall"]["obj_cnt"] for result in results])
            pred_cnt = np.sum([result["recall"]["pred_cnt"] for result in results])
            for thresh in IOU_THRESH:
                thresh_result = [result["recall"][thresh] for result in results]
                total_det = np.sum(thresh_result, axis=0)
                total_recall = total_det / obj_cnt
                agg_result[thresh] = total_recall
            avg_result = None
            for thresh in IOU_THRESH:
                thresh_result = agg_result[thresh]
                if avg_result is not None:
                    avg_result += thresh_result
                else:
                    avg_result = thresh_result.copy()
            agg_result["AR@IoUs"] = avg_result / IOU_THRESH.size
            agg_result["total_evals"] = len(results)
            agg_result["obj_cnt"] = obj_cnt
            agg_result["pred_cnt"] = pred_cnt
            agg_result["proposals"] = PROPOSALS
    return agg_result


# from classyvision
def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    tensor, orig_device = convert_to_distributed_tensor(tensor)
    gathered_tensors = [
        torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered_tensors, tensor)
    gathered_tensors = [
        _tensor.cpu() if orig_device == "cpu" else _tensor
        for _tensor in gathered_tensors
    ]
    out = torch.cat(gathered_tensors, axis=0)
    return out


def sync_result(result):
    for k in result:
        res_tensor = torch.tensor(result[k])
        res_tensor = gather_tensors_from_all(res_tensor)
        result[k] = res_tensor
    return result


def postprocess_maskrcnn_out(pred_results, has_box=False):
    coco_results = []
    for original_id, prediction in pred_results.items():
        if len(prediction) == 0:
            continue
        scores = prediction["scores"]
        masks = prediction["masks"]
        masks = masks > 0.5
        scores = prediction["scores"].tolist()

        rles = [
            mask_util.encode(
                np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
            )[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def postprocess_fasterrcnn_out(pred_results):
    coco_results = []
    for original_id, prediction in pred_results.items():
        if len(prediction) == 0:
            continue
        scores = prediction["scores"]
        boxes = prediction["boxes"]
        scores = prediction["scores"].tolist()

        # convert xyxy to xywh
        boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": 1,
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def evaluate_with_coco_protocol(
    result,
    ann_root="memcache_manifold://mscoco/tree/coco2017/annotations/instances_val2017.json",
    stats_path="manifold://weiyaowang/tree/pairwise_potential/coco_val_ids.pkl",
    eval_segm=True,
    categories=None,
    complement_category=False,
    class_agnostic=True,
):
    local_ann = ann_root
    # process ann_root
    if (categories is not None and len(categories) > 0) or complement_category:
        category_ids = pkl.load(open(stats_path, "rb"))
        if complement_category:
            new_categories = []
            if categories is None:
                categories = []
            for category in list(category_ids.keys()):
                if category not in categories:
                    new_categories.append(category)
            categories = new_categories
        print(f"evaluating {categories} categories with COCOEval")
        ids = []
        for category in categories:
            if category in category_ids:
                ids.extend(category_ids[category])
        ids = sorted(set(ids))
        coco_json = json.load(open(local_ann))
        new_image_info = []
        for image_info in coco_json["images"]:
            if image_info["id"] in ids:
                new_image_info.append(image_info)
        coco_json["images"] = new_image_info
        new_annotation = []
        for annotation in coco_json["annotations"]:
            if annotation["image_id"] not in ids:
                continue
            new_annotation.append(annotation)
            if annotation["category_id"] not in categories:
                new_annotation[-1]["ignore"] = 1
        coco_json["annotations"] = new_annotation
        new_path = local_ann[: -len(".json")] + "_temp.json"
        json.dump(coco_json, open(new_path, "w"))
        local_ann = new_path
        new_result = []
        for res in result:
            if res["image_id"] in ids:
                new_result.append(res)
        result = new_result
    cocoGt = COCO(local_ann)
    cocoDt = cocoGt.loadRes(result)
    eval_types = ["bbox"]
    if eval_segm:
        eval_types.append("segm")
    for iouType in eval_types:
        print(f"evaluate type {iouType}")
        cocoEval = COCOeval(cocoGt, cocoDt, iouType)
        maxDets = [1, 10, 100, 300, 1000]
        cocoEval.params.maxDets = maxDets
        if class_agnostic:
            cocoEval.params.useCats = 0

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # additional recall values
        s = cocoEval.eval["recall"]
        iou_cut = [0.5, 0.75, None]
        p = cocoEval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == "all"]

        for iou in iou_cut:
            for maxDet in maxDets:
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDet]
                if iou is not None:
                    t = np.where(p.iouThrs == iou)[0]
                    filtered_s = s[t]
                else:
                    filtered_s = s
                filtered_s = filtered_s[:, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(filtered_s[filtered_s > -1])
                print(f"Recall@{iou}IoU@{maxDet}Det: {mean_s}")


def rank_segments_inner(partition_label, potential, min_size):
    segment_ind = np.unique(partition_label)
    avg_map = {}
    for idx in segment_ind:
        obj_mask = partition_label == idx
        if np.sum(obj_mask) < min_size:
            continue
        obj_potential = potential * obj_mask

        avg_map[idx] = 1 - np.true_divide(
            obj_potential.sum(), (obj_potential != 0).sum()
        )

    return dict(sorted(avg_map.items(), key=lambda item: item[1]))


def rank_segments_outer(partition_label, potential, min_size):
    segment_ind = np.unique(partition_label)
    boundary = skimage.segmentation.find_boundaries(partition_label, mode="inner")
    avg_map = {}
    for idx in segment_ind:
        obj_mask = partition_label == idx
        if np.sum(obj_mask) < min_size:
            continue
        obj_potential = potential * obj_mask * boundary

        avg_map[idx] = np.true_divide(obj_potential.sum(), (obj_potential != 0).sum())

    return dict(sorted(avg_map.items(), key=lambda item: item[1]))


def compute_group_size(palette):
    labels, cnts = np.unique(palette, return_counts=True)
    size_map = {}
    for label, cnt in zip(labels, cnts):
        size_map[label] = cnt
    return size_map


def get_nms_remove_ind(size_map, obj_indices, all_segments, nms):
    remove_obj_ind = []
    for i, obj_ind in enumerate(obj_indices[1:]):
        curr_seg = all_segments[obj_ind]
        for prev_ind in obj_indices[: i + 1]:
            if prev_ind in remove_obj_ind:
                continue
            prev_seg = all_segments[prev_ind]
            intersected_seg = curr_seg & prev_seg
            union_seg = curr_seg | prev_seg
            if len(prev_seg) == 0:
                continue
            intersected_size = 0
            union_size = 0
            for seg in intersected_seg:
                intersected_size += size_map[seg]
            for seg in union_seg:
                union_size += size_map[seg]
            seg_iou = float(intersected_size) / union_size
            if seg_iou > nms:
                remove_obj_ind.append(obj_ind)
                break
    return remove_obj_ind


# filter out proposals that are possibly background
def filter_mask_by_edge(mask, method=0, thresh=0.5):
    """
    Method
    0: no filter
    1: count by # of edge with above thresh overlap
    2: edge overlap over total boundary above thresh
    """

    top_lap = np.sum(mask[0, :])
    bot_lap = np.sum(mask[-1, :])
    left_lap = np.sum(mask[:, 0])
    right_lap = np.sum(mask[:, -1])
    laps = np.array([top_lap, bot_lap, left_lap, right_lap])
    if method == 1:
        h = mask.shape[0]
        w = mask.shape[1]
        thresh_overlap = laps > (np.array([h, h, w, w]) * thresh)
        if np.sum(thresh_overlap) > 1:
            return True
    elif method == 2:
        boundary = skimage.segmentation.find_boundaries(mask, mode="inner")
        if np.sum(boundary) * thresh < np.sum(laps):
            return True
    return False


def compute_ranking_score(segment_mask, potential, rank_method):
    if rank_method == 0:
        boundary = skimage.segmentation.find_boundaries(segment_mask, mode="thick")
        obj_potential = potential * boundary
        ranked_val = np.true_divide(obj_potential.sum(), (boundary != 0).sum())
    elif rank_method == 1:
        obj_potential = potential * segment_mask
        ranked_val = -np.true_divide(obj_potential.sum(), (segment_mask != 0).sum())
    elif rank_method == 2:
        boundary = skimage.segmentation.find_boundaries(segment_mask, mode="thick")
        all_potential = potential * segment_mask
        outer_potential = potential * boundary
        potential_diff = all_potential - 2 * outer_potential
        ranked_val = -np.true_divide(potential_diff.sum(), (segment_mask != 0).sum())
    elif rank_method == 3:
        boundary = skimage.segmentation.find_boundaries(segment_mask, mode="thick")
        inner_potential = potential * (segment_mask & ~boundary)
        outer_potential = potential * boundary
        if (boundary != 0).sum() == 0 or ((segment_mask & ~boundary) != 0).sum() == 0:
            ranked_val = 100
        else:
            ranked_val = np.true_divide(outer_potential.sum(), (boundary != 0).sum())
            ranked_val -= np.true_divide(
                inner_potential.sum(), ((segment_mask & ~boundary) != 0).sum()
            )
    else:
        assert rank_method == 4, "unsupported ranking method"
        boundary = skimage.segmentation.find_boundaries(segment_mask, mode="thick")
        inner_potential = potential * (segment_mask & ~boundary)
        outer_potential = potential * boundary
        ranked_val = outer_potential.sum() / (
            inner_potential.sum() + outer_potential.sum()
        )
    return ranked_val


def proposal_ranker(
    partition_label,
    potential,
    rank_method=2,
    min_size=0,
    all_segments=None,
    nms=0,
    filter_by_edge_method=0,
    filter_by_edge_thresh=0.5,
    # customize scoring func for other use such as Box2Mask
    scoring_func=compute_ranking_score,
):
    """
    Rank all proposals based on certain property:
        0: rank by outer boundary, lower the better
        1: rank by all potential, higher the better
        2: rank by outer - inner, lower the better
        3: rank by average of outer - average of inner, lower better
        4: rank by N-Cut criteria: outer / (inner + outer)
    """
    run_nms = True
    if all_segments is None:
        all_segments = [[idx] for idx in np.unique(partition_label)]
        run_nms = False
    avg_map = {}
    mask_map = {}
    for i, segments in enumerate(all_segments):
        segment_mask = np.isin(partition_label, list(segments))
        if np.sum(segment_mask) < min_size:
            continue
        if filter_by_edge_method and filter_mask_by_edge(
            segment_mask, method=filter_by_edge_method, thresh=filter_by_edge_thresh
        ):
            continue
        mask_map[i] = segment_mask
        avg_map[i] = scoring_func(segment_mask, potential, rank_method)
    avg_map = dict(sorted(avg_map.items(), key=lambda item: item[1]))
    if run_nms and nms > 0:
        size_map = compute_group_size(partition_label)
        obj_indices = list(avg_map.keys())
        remove_obj_ind = get_nms_remove_ind(size_map, obj_indices, all_segments, nms)
        for obj_ind in remove_obj_ind:
            del avg_map[obj_ind], mask_map[obj_ind]

    return avg_map, mask_map


def postprocess_condinst_out(pred_results, class_mapper):
    """
    pred_results: {image_id: model_out(instances)}
    Assume single image
    """
    coco_results = []
    for original_id, prediction in pred_results.items():
        if len(prediction) == 0:
            continue
        if not prediction[0]["instances"].has("pred_masks"):
            continue
        scores = prediction[0]["instances"].scores.detach().cpu().numpy()
        masks = prediction[0]["instances"].pred_masks.detach().cpu().numpy()
        classes = prediction[0]["instances"].pred_classes.detach().cpu().numpy()
        rles = [
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
            )[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": class_mapper[classes[k] + 1],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def generate_pa_proposals(
    pred_pa,
    input_img=None,
    img_metas=None,
    oln_model=None,
    edge_thresh=0.1,
    join_thresh=0.9,
    rank_method=3,
    min_component_size=100,
    nms=0.5,
    filter_by_edge_method=2,
    filter_by_edge_thresh=0.2,
    use_orientation=0,
    use_globalization=0.0,
    use_new_affinity=True,
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
    rank_map, mask_map = proposal_ranker(
        pred_palette,
        max_affinity,
        rank_method=rank_method,
        min_size=min_component_size,
        all_segments=all_segments,
        nms=nms,
        filter_by_edge_method=filter_by_edge_method,
        filter_by_edge_thresh=filter_by_edge_thresh,
    )
    masks = []
    scores = []
    for label_idx, avg_pp in rank_map.items():
        masks.append(mask_map[label_idx])
        scores.append(1 - avg_pp)
    if len(masks) == 0:
        print(f"empty masks: {np.unique(pred_palette)}")
        return None, None
    return masks, scores
    # return np.stack(masks)[:, np.newaxis, :, :], np.array(scores)
