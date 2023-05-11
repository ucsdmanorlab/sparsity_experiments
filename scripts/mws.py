import numpy as np

from funlib.persistence import open_ds
from funlib.geometry import Coordinate,Roi

from skimage.filters import threshold_otsu

import mahotas
from affogato.segmentation import compute_mws_segmentation
from affogato.segmentation import MWSGridGraph, compute_mws_clustering
from typing import Optional

from skimage.morphology import remove_small_objects
from scipy.ndimage import distance_transform_edt, maximum_filter, binary_erosion
from skimage.measure import label


def erode(labels, steps, only_xy=True):
    
    if only_xy:
        assert len(labels.shape) == 3
        for z in range(labels.shape[0]):
            labels[z] = erode(labels[z], steps, only_xy=False)
        return labels

    # get all foreground voxels by erosion of each component
    foreground = np.zeros(shape=labels.shape, dtype=bool)
    
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = labels==label
        # Assume that masked out values are the same as the label we are
        # eroding in this iteration. This ensures that at the boundary to
        # a masked region the value blob is not shrinking.
        eroded_label_mask = binary_erosion(label_mask, iterations=steps, border_value=1)

        foreground = np.logical_or(eroded_label_mask, foreground)

    # label new background
    background = np.logical_not(foreground)
    labels[background] = 0

    return labels


def expand_labels(labels):

    distance = labels.shape[0]

    distances, indices = distance_transform_edt(
            labels == 0,
            return_indices=True)

    expanded_labels = np.zeros_like(labels)

    dilate_mask = distances <= distance

    masked_indices = [
            dimension_indices[dilate_mask]
            for dimension_indices in indices
    ]

    nearest_labels = labels[tuple(masked_indices)]

    expanded_labels[dilate_mask] = nearest_labels

    return expanded_labels


def watershed_from_boundary_distance(
    boundary_distances, return_seeds=False, id_offset=0, min_seed_distance=10
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = mahotas.cwatershed(boundary_distances.max() - boundary_distances, seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def mutex_watershed(
        affs,
        offsets,
        stride,
        algorithm="kruskal",
        mask=None,
        randomize_strides=True,
        sep=3) -> np.ndarray:

    affs = 1 - affs

    affs[:sep] = affs[:sep] * -1
    affs[:sep] = affs[:sep] + 1

    segmentation = compute_mws_segmentation(
        affs,
        offsets,
        sep,
        strides=stride,
        randomize_strides=randomize_strides,
        algorithm=algorithm,
        mask=mask,
    )

    return segmentation


def seeded_mutex_watershed(
        seeds,
        affs,
        offsets,
        mask,
        stride,
        randomize_strides=True) -> np.ndarray:
    
    shape = affs.shape[1:]
    if seeds is not None:
        assert (len(seeds.shape) == len(shape)
        ), f"Got shape {seeds.data.shape} for mask but expected {shape}"
    if mask is not None:
        assert (len(mask.shape) == len(shape)
        ), f"Got shape {mask.data.shape} for mask but expected {shape}"

    grid_graph = MWSGridGraph(shape)
    if seeds is not None:
        grid_graph.update_seeds(seeds.data)

    ndim = len(offsets[0])

    grid_graph.add_attractive_seed_edges = True
    neighbor_affs, lr_affs = (
        np.require(affs[:ndim], requirements="C"),
        np.require(affs[ndim:], requirements="C"),
    )
    
    # assuming affinities are 1 between voxels that belong together and
    # 0 if they are not part of the same object. Invert if the other way
    # around.
    # neighbors_affs should be high for objects that belong together
    # lr_affs is the oposite
    lr_affs = 1 - lr_affs

    uvs, weights = grid_graph.compute_nh_and_weights(
        neighbor_affs, offsets[:ndim]
    )

    if stride is None:
        stride = [1] * ndim
        
    grid_graph.add_attractive_seed_edges = False
    mutex_uvs, mutex_weights = grid_graph.compute_nh_and_weights(
        lr_affs,
        offsets[ndim:],
        stride,
        randomize_strides=randomize_strides,
    )

    # compute the segmentation
    n_nodes = grid_graph.n_nodes
    segmentation = compute_mws_clustering(
        n_nodes, uvs, mutex_uvs, weights, mutex_weights
    )
    grid_graph.relabel_to_seeds(segmentation)
    segmentation = segmentation.reshape(shape)
    if mask is not None:
        segmentation[np.logical_not(mask)] = 0

    return segmentation


def post(
        pred_file,
        pred_dataset,
        roi,
        normalize_preds,
        neighborhood,
        stride,
        randomize_strides,
        algorithm,
        mask_thresh,
        erode_steps,
        clean_up):

    # load
    pred = open_ds(pred_file,pred_dataset)

    if roi is not None:
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    pred = pred.to_ndarray(roi)
    
    # normalize
    pred = (pred / np.max(pred)).astype(np.float32)
    
    # normalize channel-wise
    if normalize_preds:
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
    
    # prepare
    neighborhood = [tuple(x) for x in neighborhood]
    
    if mask_thresh > 0.0 or algorithm == "seeded":    
        
        mean_pred = 0.5 * (pred[1] + pred[2])
        depth = mean_pred.shape[0]

        if mask_thresh > 0.0:
            mask = np.zeros(mean_pred.shape, dtype=bool)
        
        if algorithm == "seeded":
            seeds = np.zeros(mean_pred.shape, dtype=np.uint64)

        for z in range(depth):

            boundary_mask = mean_pred[z] > mask_thresh * np.max(pred)
            boundary_distances = distance_transform_edt(boundary_mask)
            if mask_thresh > 0.0:
                mask[z] = boundary_mask

            if algorithm == "seeded":
                _,_,seeds[z] = watershed_from_boundary_distance(
                    boundary_distances,
                    return_seeds=True,
                )
    
    if mask_thresh == 0.0:
        mask = None

    if "seeded" in algorithm:
        seeds = seeds if "wo" not in algorithm else None   
        
        seg = seeded_mutex_watershed(
            seeds=seeds,
            affs=pred,
            offsets=neighborhood,
            mask=mask,
            stride=stride,
            randomize_strides=randomize_strides)
    
    else:
        seg = mutex_watershed(
            pred,
            offsets=neighborhood,
            stride=stride,
            algorithm=algorithm,
            mask=mask,
            randomize_strides=randomize_strides)

    # clean up
    if clean_up > 0:
        seg = remove_small_objects(seg.astype(np.int64),min_size=clean_up)
        seg = expand_labels(seg)
        seg = label(seg, connectivity=1).astype(np.uint64)

    # erode
    if erode_steps > 0:
        seg = erode(seg,erode_steps)
    
    return seg
