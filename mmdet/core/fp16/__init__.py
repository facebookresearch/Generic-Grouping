# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from .deprecated_fp16_utils import (
    deprecated_auto_fp16 as auto_fp16,
    deprecated_force_fp32 as force_fp32,
    deprecated_wrap_fp16_model as wrap_fp16_model,
    DeprecatedFp16OptimizerHook as Fp16OptimizerHook,
)

__all__ = ["auto_fp16", "force_fp32", "Fp16OptimizerHook", "wrap_fp16_model"]
