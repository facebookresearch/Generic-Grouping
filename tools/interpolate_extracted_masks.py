import json
import multiprocessing as mp
import time

import cv2
import numpy as np
from pycocotools import mask as maskUtils


MASKS_DIR = ""
NUM_SPLITS = 1


def resize_mask(image_ann):
    new_anns = []
    for ann in image_ann:
        segm = ann["segmentation"]
        mask = maskUtils.decode(segm)
        orig_shape = ann["ori_shape"][:2]
        resized_mask = cv2.resize(
            mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
        )
        new_rle = maskUtils.encode(np.asfortranarray(resized_mask))
        if type(new_rle["counts"]) == bytes:
            new_rle["counts"] = new_rle["counts"].decode("ascii")
        area = maskUtils.area(new_rle)
        bbox = maskUtils.toBbox(new_rle)
        ann["segmentation"] = new_rle
        ann["area"] = int(area)
        ann["bbox"] = [int(coord) for coord in bbox]
        new_anns.append(ann)
    return new_anns


SPLITS = range(0, NUM_SPLITS)
for split in SPLITS:
    json_path = f"{MASKS_DIR}/masks_{split}.json"
    output_path = f"{MASKS_DIR}/masks_interpolated_{split}.json"

    ann_json = json.load(open(json_path, "rb"))

    start = time.perf_counter()

    mp_pool = mp.Pool(processes=60)
    resized_masks = mp_pool.map(resize_mask, ann_json)

    print(f"finished {split}")
    print(time.perf_counter() - start, "seconds")

    json.dump(resized_masks, open(output_path, "w"))
