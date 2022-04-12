# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import json
import random

from pycocotools.coco import COCO

from .builder import DATASETS
from .coco_split import CocoSplitDataset


@DATASETS.register_module()
class CocoSplitOnlineDataset(CocoSplitDataset):
    """
    Different from other MMDet dataset, this one loads annotations
        online instead of from a whole json. This is more memory
        efficient albeit a little bit slower.
    This enables training on 3M+ masks, which would not be feasible
        with a single json to store.
    """

    def __init__(
        self,
        ann_dir=None,
        iou_thresh=None,
        score_thresh=None,
        top_k=None,
        random_sample_masks=False,
        **kwargs,
    ):
        """
        Args:
            ann_dir: directory to store the annotations, where annotations
                for each image is stored as "image_id.json"
        For other arguments please see coco_split_pseudo_masks.py
        """
        self.ann_dir = ann_dir
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.top_k = top_k
        self.random_sample_masks = random_sample_masks
        super(CocoSplitOnlineDataset, self).__init__(**kwargs)

    # Override to load pseudo masks online
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]["id"]
        ann_info = json.load(open(f"{self.ann_dir}{img_id}.json"))
        ann_info = self.sample_targets(ann_info)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def sample_targets(self, annotations):
        new_anns = annotations
        if self.iou_thresh is not None:
            tmp_new_anns = []
            for ann in new_anns:
                if ann["gt_iou"] < self.iou_thresh:
                    tmp_new_anns.append(ann)
            new_anns = tmp_new_anns
        if self.score_thresh is not None:
            tmp_new_anns = []
            for ann in new_anns:
                if ann["score"] >= self.score_thresh:
                    tmp_new_anns.append(ann)
            new_anns = tmp_new_anns
        if self.random_sample_masks:
            random.shuffle(new_anns)
        if self.top_k is not None:
            new_anns = new_anns[: self.top_k]
        return new_anns
