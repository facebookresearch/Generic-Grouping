# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


# cython: language_level=3

cimport cython
cimport numpy as np
from libc.stdint cimport int32_t, int64_t

import numpy as np

cdef int64_t loc2idx(
    int32_t i,
    int32_t j,
    int32_t width,
):
    cdef int64_t index = i * width + j
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef list generate_nodes_edges_labels_c(
    np.ndarray[np.float32_t, ndim=3] potentials,
):
    cdef int32_t height = potentials.shape[1]
    cdef int32_t width = potentials.shape[2]
    cdef np.ndarray[np.uint32_t, ndim=2] new_label

    cdef int64_t num_nodes = int(potentials.size / 4)

    cdef list out_nodes = []
    cdef dict property
    cdef tuple node
    for i in range(num_nodes):
        property = {"labels": [i]}
        node = (i, property)
        out_nodes.append(node)

    cdef np.ndarray[np.float32_t, ndim=1] potential
    cdef int64_t curr_idx, neighbor
    cdef list out_edges = []
    cdef tuple edge

    new_label = np.zeros((height, width), dtype=np.uint32)

    for i in range(height):
        for j in range(width):
            potential = potentials[:, i, j]
            curr_idx = loc2idx(i, j, width)
            new_label[i, j] = curr_idx
            if i - 1 >= 0:
                neighbor = loc2idx(i - 1, j, width)
                edge = (curr_idx, neighbor, 1 - potential[0])
                out_edges.append(edge)
            if j - 1 >= 0:
                neighbor = loc2idx(i, j - 1, width)
                edge = (curr_idx, neighbor, 1 - potential[1])
                out_edges.append(edge)
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor = loc2idx(i - 1, j - 1, width)
                edge = (curr_idx, neighbor, 1 - potential[2])
                out_edges.append(edge)
            if i + 1 < height and j - 1 >= 0:
                neighbor = loc2idx(i + 1, j - 1, width)
                edge = (curr_idx, neighbor, 1 - potential[3])
                out_edges.append(edge)
    return [out_nodes, out_edges, new_label]


def generate_nodes_edges_labels(
    np.ndarray potentials,
) -> list:
    return generate_nodes_edges_labels_c(potentials)
