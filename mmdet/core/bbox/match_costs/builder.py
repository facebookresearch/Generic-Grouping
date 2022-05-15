# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from mmcv.utils import build_from_cfg, Registry

MATCH_COST = Registry("Match Cost")


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, MATCH_COST, default_args)
