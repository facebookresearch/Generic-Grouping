# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from .generic_roi_extractor import GenericRoIExtractor
from .single_level_roi_extractor import SingleRoIExtractor

__all__ = [
    "SingleRoIExtractor",
    "GenericRoIExtractor",
]
