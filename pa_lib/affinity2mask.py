# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


import cv2
import networkx as nx
import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from scipy import sparse
from skimage.feature import peak_local_max
from skimage.future import graph as sk_graph
from skimage.segmentation import watershed

from .cython_lib.cython_lib.graph_helper import generate_nodes_edges_labels
from .rag import rag_boundary, merge_hierarchical


def loc2idx(loc, height, width):
    index = loc[0] * width + loc[1]
    return index


def expand_potential(potentials):
    new_potential = np.zeros_like(potentials)
    new_potential[0, :-1, :] = potentials[0, 1:, :]
    new_potential[1, :, :-1] = potentials[1, :, 1:]
    new_potential[2, :-1, :-1] = potentials[2, 1:, 1:]
    new_potential[3, 1:, :-1] = potentials[3, :-1, 1:]
    return np.concatenate([potentials, new_potential], axis=0)


def potentials2graph(potentials):
    graph = nx.Graph()
    height = potentials.shape[1]
    width = potentials.shape[2]
    new_label = np.zeros_like(potentials[0, :, :], dtype=np.uint32)
    num_nodes = int(potentials.size / 4)
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        graph.nodes[i].update({"labels": [i]})
    for i in range(height):
        for j in range(width):
            potential = potentials[:, i, j]
            curr_idx = loc2idx([i, j], height, width)
            new_label[i, j] = curr_idx
            if i - 1 >= 0:
                neighbor = loc2idx([i - 1, j], height, width)
                graph.add_edge(curr_idx, neighbor, weight=1 - potential[0])
            if j - 1 >= 0:
                neighbor = loc2idx([i, j - 1], height, width)
                graph.add_edge(curr_idx, neighbor, weight=1 - potential[1])
            if i - 1 >= 0 and j - 1 >= 0:
                neighbor = loc2idx([i - 1, j - 1], height, width)
                graph.add_edge(curr_idx, neighbor, weight=1 - potential[2])
            if i + 1 < height and j - 1 >= 0:
                neighbor = loc2idx([i + 1, j - 1], height, width)
                graph.add_edge(curr_idx, neighbor, weight=1 - potential[3])
    return sk_graph.RAG(data=graph), new_label


def potentials2graph_cython(potentials):
    graph = nx.Graph()
    [nodes, edges, labels] = generate_nodes_edges_labels(potentials)
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(edges)
    return sk_graph.RAG(data=graph), labels


def graph2palette(
    graph,
    label,
    edge_threshold,
    min_component_size,
):
    raw_palette = sk_graph.cut_threshold(label, graph, thresh=edge_threshold)
    raw_palette += 1  # get rid of label 0
    idx, cnts = np.unique(raw_palette, return_counts=True)
    filtered_idx = idx[cnts >= min_component_size]
    filtered_palette = np.multiply(np.isin(raw_palette, filtered_idx), raw_palette)
    return filtered_palette


def palette2masks(palette):
    mask_indices = np.unique(palette)
    all_masks = []
    for idx in mask_indices:
        if idx == 0:
            continue
        mask = palette == idx
        all_masks.append(mask)
    return all_masks


def convert_8neighbors_potentials(pred_potentials):
    pred_potentials = np.zeros((4,) + pred_potentials.shape[1:])
    pred_potentials[0, 1:, :] = np.maximum(
        pred_potentials[0, 1:, :], pred_potentials[4, :-1, :]
    )
    pred_potentials[1, :, 1:] = np.maximum(
        pred_potentials[1, :, 1:], pred_potentials[5, :, :-1]
    )
    pred_potentials[2, 1:, 1:] = np.maximum(
        pred_potentials[2, 1:, 1:], pred_potentials[6, :-1, :-1]
    )
    pred_potentials[3, :-1, 1:] = np.maximum(
        pred_potentials[3, :-1, 1:], pred_potentials[7, 1:, :-1]
    )
    return pred_potentials


# connected component
def potential2masks_cc(
    pred_potentials,
    edge_thresholds,
    min_component_size,
    use_cython=True,
    output_palette=False,
):
    if pred_potentials.shape[0] == 8:
        pred_potentials = convert_8neighbors_potentials(pred_potentials)
    if use_cython:
        graph, label = potentials2graph_cython(pred_potentials)
    else:
        graph, label = potentials2graph(pred_potentials)
    all_masks = []
    for thresh in edge_thresholds:
        tmp_graph = graph.copy()
        palette = graph2palette(tmp_graph, label, thresh, min_component_size)
        if output_palette:
            return palette
        masks = palette2masks(palette)
        all_masks.extend(masks)
        del tmp_graph
    return all_masks


def potential2masks_watershed(
    max_potential,
    basin_threshold,
    output_palette=False,
):
    _, sure_fg = cv2.threshold(
        max_potential, basin_threshold * max_potential.max(), 1, 0
    )
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[sure_fg == 0] = 0
    labels = watershed(-max_potential, markers, mask=sure_fg == 0)
    if output_palette:
        return labels
    return palette2masks(labels)


def construct_finest_partition_watershed(
    max_potential,
    local_max_threshold,
    local_peak_footprint_size=20,
):
    # generate pre-filter with foreground likelihood
    _, sure_fg = cv2.threshold(
        max_potential, local_max_threshold * max_potential.max(), 1, 0
    )
    # generate local max
    if type(local_peak_footprint_size) == int:
        footprint = np.ones((local_peak_footprint_size, local_peak_footprint_size))
    else:
        footprint = np.ones(local_peak_footprint_size)
    coords = peak_local_max(
        max_potential,
        footprint=footprint,
    )
    mask = np.zeros(max_potential.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, num_feat = ndi.label(mask)
    markers = markers * sure_fg

    if len(np.unique(markers)) < 2:
        # print(coords)
        # print(np.unique(markers))
        print(
            f"less than 2 markers provided: {np.unique(markers)}, {np.sum(sure_fg)}, {num_feat}"
        )
        return None

    labels = watershed(-max_potential, markers, connectivity=2)
    return labels


def hierarchy2segments(rag_graph):
    """
    Turn a hierarchical segmented rag into segments
    The segments are List[Set], where each segment list
        contains init labels from over-segmentation
    """
    all_nodes = []
    segments = []
    all_nodes.extend(rag_graph.removed_nodes)
    for node in rag_graph.nodes:
        all_nodes.append(rag_graph.nodes[node])
    for node in all_nodes:
        segments.append(set(node["labels"]))
    return segments


def triangle_kernel(kerlen):
    r = np.arange(kerlen)
    kernel1d = (kerlen + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d /= kernel2d.sum()
    return kernel2d


def compute_edge_orientations(edge_map):
    tri_filters = triangle_kernel(9)
    boundaries_tri_filtered = scipy.signal.convolve2d(edge_map, tri_filters, "same")
    delta_kernel = np.array([[-1, 2, -1]])
    Dy = scipy.signal.convolve2d(boundaries_tri_filtered, delta_kernel, "same")
    Dx = scipy.signal.convolve2d(
        boundaries_tri_filtered, delta_kernel.transpose(), "same"
    )
    Dy_filter_kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    Dy_filter = (
        scipy.signal.convolve2d(boundaries_tri_filtered, Dy_filter_kernel, "same") > 0
    )
    Dy[Dy_filter] = -Dy[Dy_filter]
    orientation = np.mod(np.arctan2(Dy, Dx), np.pi)
    return orientation


def construct_affinity_matrix(pa):
    max_idx = pa.size
    total_affinities = 5 * max_idx - 2 * pa.shape[0] - 2 * pa.shape[1]
    data = np.zeros(total_affinities)
    row = np.zeros(total_affinities)
    col = np.zeros(total_affinities)

    # construct identity
    data[:max_idx] = np.ones(max_idx)
    row[:max_idx] = np.arange(max_idx)
    col[:max_idx] = np.arange(max_idx)
    data_ptr = max_idx

    orig_idx = np.arange(max_idx).reshape(pa.shape)

    # horizontal neighbors
    horizontal_boundary = np.minimum(pa[:, 1:], pa[:, :-1])
    data[data_ptr : data_ptr + horizontal_boundary.size] = horizontal_boundary.flatten()
    row[data_ptr : data_ptr + horizontal_boundary.size] = orig_idx[:, 1:].flatten()
    col[data_ptr : data_ptr + horizontal_boundary.size] = orig_idx[:, :-1].flatten()

    data_ptr = data_ptr + horizontal_boundary.size
    data[data_ptr : data_ptr + horizontal_boundary.size] = horizontal_boundary.flatten()
    col[data_ptr : data_ptr + horizontal_boundary.size] = orig_idx[:, 1:].flatten()
    row[data_ptr : data_ptr + horizontal_boundary.size] = orig_idx[:, :-1].flatten()

    data_ptr = data_ptr + horizontal_boundary.size
    # vertical neighbors
    vertical_boundary = np.minimum(pa[1:], pa[:-1])
    data[data_ptr : data_ptr + vertical_boundary.size] = vertical_boundary.flatten()
    col[data_ptr : data_ptr + vertical_boundary.size] = orig_idx[1:].flatten()
    row[data_ptr : data_ptr + vertical_boundary.size] = orig_idx[:-1].flatten()
    data_ptr = data_ptr + vertical_boundary.size

    data[data_ptr : data_ptr + vertical_boundary.size] = vertical_boundary.flatten()
    row[data_ptr : data_ptr + vertical_boundary.size] = orig_idx[1:].flatten()
    col[data_ptr : data_ptr + vertical_boundary.size] = orig_idx[:-1].flatten()

    affinity_matrix = sparse.csr_matrix(
        (data, (row, col)), shape=(max_idx, max_idx), dtype=np.float
    )
    return affinity_matrix


def construct_diagonal_matrix(pa):
    summary = np.ones_like(pa)
    summary[1:] += np.minimum(pa[1:], pa[:-1])
    summary[:-1] += np.minimum(pa[1:], pa[:-1])
    summary[:, 1:] += np.minimum(pa[:, 1:], pa[:, :-1])
    summary[:, :-1] += np.minimum(pa[:, 1:], pa[:, :-1])
    max_idx = pa.size
    diagonal_matrix = sparse.csr_matrix(
        (summary.flatten(), (np.arange(max_idx), np.arange(max_idx))),
        shape=(max_idx, max_idx),
        dtype=np.float,
    )
    return diagonal_matrix


def eigenvec2edge(vecs, vals, img_shape):
    # remove first eigenvalue
    vecs = vecs[:, 1:]
    vals = vals[1:]

    normalized_vec = (vecs - vecs.min(axis=0)) / (vecs.max(axis=0) - vecs.min(axis=0))
    normalized_vec = normalized_vec.reshape(img_shape + (vecs.shape[1],))
    normalized_vec = normalized_vec / np.sqrt(vals)
    vertical_boundary = np.abs(normalized_vec[:, :-1] - normalized_vec[:, 1:])
    horizontal_boundary = np.abs(normalized_vec[:-1, :] - normalized_vec[1:, :])
    new_boundary = np.zeros_like(normalized_vec)

    new_boundary[1:] = np.maximum(horizontal_boundary, new_boundary[1:])
    new_boundary[:-1] = np.maximum(horizontal_boundary, new_boundary[:-1])

    new_boundary[:, :-1] = np.maximum(vertical_boundary, new_boundary[:, :-1])
    new_boundary[:, 1:] = np.maximum(vertical_boundary, new_boundary[:, 1:])

    new_boundary = np.power(new_boundary.sum(axis=2), 1 / np.sqrt(2))
    return new_boundary


def get_globalized_boundary(boundary_prob, nvec=16, affinity_gamma=0.12):
    local_affinity = np.exp(-boundary_prob / affinity_gamma)
    affinity_matrix = construct_affinity_matrix(local_affinity)
    diagonal_matrix = construct_diagonal_matrix(local_affinity)
    vals, vecs = sparse.linalg.eigsh(
        A=diagonal_matrix - affinity_matrix,
        M=diagonal_matrix,
        k=nvec,
        which="LM",
        sigma=0,
    )
    global_boundary = eigenvec2edge(vecs, vals, local_affinity.shape)
    return global_boundary


def potential2masks_ucm_hierarchy(
    max_potential,
    local_max_threshold,
    merge_threshold,
    footprint_size=20,
    use_orientation=False,
    use_globalization=0.0,
):
    init_labels = construct_finest_partition_watershed(
        max_potential,
        local_max_threshold,
        local_peak_footprint_size=footprint_size,
    )
    if init_labels is None:
        return None, None
    new_potential = max_potential
    wt_boundaries = skimage.segmentation.find_boundaries(init_labels)
    if use_orientation:
        wt_orientation = compute_edge_orientations(wt_boundaries)
        pa_boundaries = 1 - new_potential
        pa_orientations = compute_edge_orientations(pa_boundaries)
        orientation_alignment = np.abs(np.cos((wt_orientation - pa_orientations)))
        new_potential = 1 - orientation_alignment * wt_boundaries * pa_boundaries
    if use_globalization:
        local_boundary = 1 - new_potential
        global_boundary = get_globalized_boundary(1 - new_potential)
        global_boundary = global_boundary / global_boundary.max()
        combined_boundary = (
            1.0 - use_globalization
        ) * local_boundary + global_boundary * use_globalization
        new_potential = 1 - combined_boundary
    rag_graph = rag_boundary(init_labels, 1 - new_potential)
    new_potential = wt_boundaries * new_potential + (1 - wt_boundaries) * max_potential
    if rag_graph is None:
        print(f"None rag graph, {np.unique(init_labels)}")
        return None, None, None
    _, merged_rag = merge_hierarchical(init_labels, rag_graph, merge_threshold)
    all_segments = hierarchy2segments(merged_rag)
    return init_labels, all_segments, new_potential


def potential2masks_fallback_connected_component(
    max_potential,
    fall_back_connect_thresh=0.85,
):
    _, sure_fg = cv2.threshold(
        max_potential, fall_back_connect_thresh * max_potential.max(), 1, 0
    )
    return sure_fg
