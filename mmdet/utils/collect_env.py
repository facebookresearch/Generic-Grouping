# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import mmdet
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info["MMDetection"] = mmdet.__version__ + "+" + get_git_hash()[:7]
    return env_info


if __name__ == "__main__":
    for name, val in collect_env().items():
        print(f"{name}: {val}")
