import sys
import os
import numpy as np

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate,Roi

from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion
from skimage.measure import label
from skimage.morphology import remove_small_objects
import mahotas
import waterz
import zarr


waterz_merge_function = {
    'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
    'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
    'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
    'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
    'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
    'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
    'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
    'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
    'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
    'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
    'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
}


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


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=10,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances, return_seeds, min_seed_distance=min_seed_distance
        )

        fragments = ret[0]

    return ret


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


def post(
        pred_file,
        pred_dataset,
        roi=None,
        normalize_preds=False,
        min_seed_distance=10,
        merge_function="mean",
        thresholds=None,
        erode_steps=0,
        clean_up=0,
        **kwargs):

    # load
    pred = open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    # first three channels are direct neighbor affs
    pred = pred.to_ndarray(roi)[:3]
    
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
   
    # watershed
    fragments = watershed_from_affinities(
        pred,
        fragments_in_xy=True,
        min_seed_distance=min_seed_distance)[0]
    
    # agglomerate
    max_thresh = 1.0
    step = 1/20
   
    if thresholds is None:
        thresholds = [round(x,2) for x in np.arange(0,max_thresh,step)]

    segs = {}

    generator = waterz.agglomerate(
            pred,
            thresholds=thresholds,
            fragments=fragments.copy(),
            scoring_function=waterz_merge_function[merge_function])

    for threshold,segmentation in zip(thresholds,generator):
       
        #seg = segmentation.copy()
#
#        # clean
#        if clean_up > 0:
#            seg = remove_small_objects(seg.astype(np.int64),min_size=clean_up)
#            seg = expand_labels(seg)
#            seg = label(seg, connectivity=1)
#
#        # erode
#        if erode_steps > 0:
#            seg = erode(seg,erode_steps)

        segs[threshold] = segmentation.copy()#mentation#.astype(np.uint64)

    return segs, fragments

if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    out_file = sys.argv[3]
    out_ds = sys.argv[4]
    roi = None
    thresh = sys.argv[5]
    merge_fn = sys.argv[6]
    norm = int(sys.argv[7])

    segs,_ = post(
            pred_file,
            pred_dataset,
            roi=roi,
            normalize_preds=bool(norm),
            merge_function=merge_fn,
            thresholds=[thresh])
    
    pred = open_ds(pred_file,pred_dataset)

    if roi is not None:
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi
    
    print("writing")

    out_seg = prepare_ds(
            out_file,
            out_ds,
            roi,
            pred.voxel_size,
            np.uint64,
            delete=True)

    out_seg[roi] = segs[float(thresh)]
