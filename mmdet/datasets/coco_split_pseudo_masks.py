import random

from pycocotools.coco import COCO

from .builder import DATASETS
from .coco_split import CocoSplitDataset


@DATASETS.register_module()
class CocoSplitPseudoMasksDataset(CocoSplitDataset):
    """
    Used to joint train on images with both pseudo-GT and GT.
    """

    def __init__(
        self,
        additional_ann_file=None,
        iou_thresh=None,
        score_thresh=None,
        top_k=None,
        random_sample_masks=False,
        **kwargs,
    ):
        # Add additional annotation file (eg. from pseudo masks)
        self.additional_coco = None
        if additional_ann_file is not None:
            self.additional_coco = COCO(additional_ann_file)
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.top_k = top_k
        self.random_sample_masks = random_sample_masks
        super(CocoSplitPseudoMasksDataset, self).__init__(**kwargs)

    # Override to load pseudo masks
    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]["id"]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        all_anns = []
        all_anns.extend(ann_info)
        if self.additional_coco is not None:
            additional_ann_ids = self.additional_coco.get_ann_ids(img_ids=[img_id])
            additional_ann_info = self.additional_coco.load_anns(additional_ann_ids)
            additional_ann_info = self.sample_targets(additional_ann_info)
            all_anns.extend(additional_ann_info)
        return self._parse_ann_info(self.data_infos[idx], all_anns)

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
