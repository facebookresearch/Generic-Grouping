# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from mmdet import digit_version


def test_version_check():
    assert digit_version("1.0.5") > digit_version("1.0.5rc0")
    assert digit_version("1.0.5") > digit_version("1.0.4rc0")
    assert digit_version("1.0.5") > digit_version("1.0rc0")
    assert digit_version("1.0.0") > digit_version("0.6.2")
    assert digit_version("1.0.0") > digit_version("0.2.16")
    assert digit_version("1.0.5rc0") > digit_version("1.0.0rc0")
    assert digit_version("1.0.0rc1") > digit_version("1.0.0rc0")
    assert digit_version("1.0.0rc2") > digit_version("1.0.0rc0")
    assert digit_version("1.0.0rc2") > digit_version("1.0.0rc1")
    assert digit_version("1.0.1rc1") > digit_version("1.0.0rc1")
    assert digit_version("1.0.0") > digit_version("1.0.0rc1")
