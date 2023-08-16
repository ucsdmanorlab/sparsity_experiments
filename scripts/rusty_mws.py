import sys
import os
import numpy as np

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate,Roi
from funlib.segment.arrays import replace_values, relabel

from scipy.ndimage import measurements, gaussian_filter

import mwatershed as mws

import zarr


offsets = [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-2, 0, 0],
        [0, -2, 0],
        [0, 0, -2],
        [-3, 0, 0],
        [0, -4, 0],
        [0, 0, -4],
        [-4, 0, 0],
        [0, -8, 0],
        [0, 0, -8]
    ]

context = [12,40,40]


def segment(
        pred_file,
        pred_dataset,
        #offsets,
        roi=None,
        normalize_preds=False,
        lr_bias_ratio=-0.175,
        adjacent_edge_bias=0.4,
        filter_val=0.0,
        seeds_file=None,
        seeds_dataset=None,
        **kwargs):

    # load
    pred = open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    # first three channels are direct neighbor affs
    pred = pred.to_ndarray(roi).astype(np.float64)/255.0
    
    # normalize
    pred = (pred / np.max(pred))

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.
    random_noise: float = np.random.randn(*pred.shape) * 0.001

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.
    smoothed_affs: np.ndarray = (
        gaussian_filter(pred, sigma=(0, *(Coordinate(context) / 3)))
        - 0.5
    ) * 0.01

    shift: np.ndarray = np.array(
        [
            #adjacent_edge_bias if max(offset) <= 1
            adjacent_edge_bias if abs(min(offset)) <= 1
            # else lr_edge_bias
            else np.linalg.norm(offset) * lr_bias_ratio
            for offset in offsets
        ]
    ).reshape((-1, *((1,) * (len(pred.shape) - 1))))

    # normalize channel-wise
    if normalize_preds:
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
   
    if seeds_file is not None and seeds_dataset is not None:
        seeds = open_ds(seeds_file,seeds_dataset).to_ndarray(roi)
    else:
        seeds = None

    seg = mws.agglom(
            pred + shift + random_noise + smoothed_affs,
            offsets,
            seeds)

    # filter fragments
    if filter_val > 0.0:
        average_affs: float = np.mean(pred, axis=0)

        filtered_fragments: list = []

        fragment_ids: np.ndarray = np.unique(seg)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, seg, fragment_ids)
        ):
            if mean < filter_val:
                filtered_fragments.append(fragment)

        filtered_fragments: np.ndarray = np.array(filtered_fragments, dtype=seg.dtype)
        replace: np.ndarray = np.zeros_like(filtered_fragments)
        replace_values(seg, filtered_fragments, replace, inplace=True)

    # relabel
    seg, max_id = relabel(seg)

    return seg


if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    out_file = sys.argv[3]
    out_ds = sys.argv[4]
    roi = None
    norm = int(sys.argv[5])
    lr_bias_ratio = float(sys.argv[6])
    adjacent_edge_bias = float(sys.argv[7])  # bias towards merging
    filter_val = float(sys.argv[8])

    try:
        seeds_file = sys.argv[9]
        seeds_dataset = sys.argv[10]
    except:
        seeds_file = None
        seeds_dataset = None

    seg = segment(
            pred_file,
            pred_dataset,
            #offsets,
            roi=roi,
            normalize_preds=bool(norm),
            lr_bias_ratio=lr_bias_ratio,
            adjacent_edge_bias=adjacent_edge_bias,
            filter_val=filter_val,
            seeds_file=seeds_file,
            seeds_dataset=seeds_dataset)
    
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
            np.uint64)

    out_seg[roi] = seg
