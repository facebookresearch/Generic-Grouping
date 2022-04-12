# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


# Copyright (c) Open-MMLab. All rights reserved.

__version__ = "2.8.0"
short_version = __version__


def parse_version_info(version_str):
    version_info = []
    for x in version_str.split("."):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find("rc") != -1:
            patch_version = x.split("rc")
            version_info.append(int(patch_version[0]))
            version_info.append(f"rc{patch_version[1]}")
    return tuple(version_info)


version_info = parse_version_info(__version__)
