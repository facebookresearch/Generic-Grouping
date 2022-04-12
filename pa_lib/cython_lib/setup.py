# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [
    Extension(r"cython_lib.graph_helper", [r"graph_helper.pyx"]),
]

setup(
    name="cython_lib",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
