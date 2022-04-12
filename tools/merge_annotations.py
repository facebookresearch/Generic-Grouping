# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import json

splits = 4

# COCO
pseudo_mask_path = "WORK_DIR/masks_{}.json"
ref_json_path = "DATA_DIR/instances.json"

output_path = pseudo_mask_path.format("all")

# JOIN into processed
ref_json = json.load(open(ref_json_path, "rb"))
max_id = 0
for ann in ref_json["annotations"]:
    max_id = max(ann["id"], max_id)

pseudo_masks = []
for shard in range(splits):
    ann_json = json.load(open(pseudo_mask_path.format(shard), "rb"))
    for img_ann in ann_json:
        for ann in img_ann:
            max_id += 1
            ann["id"] = max_id
            pseudo_masks.append(ann)

ref_json["annotations"] = pseudo_masks

print(f"generated {len(pseudo_masks)} pseudo masks")
json.dump(ref_json, open(output_path, "w"))
