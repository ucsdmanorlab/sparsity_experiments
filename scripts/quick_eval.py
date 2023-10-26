import sys
import json
import os
import multiprocessing
import tqdm
import itertools
#import gc

import numpy as np
from funlib.persistence import open_ds
from funlib.geometry import Coordinate,Roi
from funlib.evaluate import rand_voi
import hierarchical


def evaluate(
    seg,
    labels,
    mask=None,
    thresh=None):

    #ensure same shape
    if mask is not None:
        if seg.shape != mask.shape:
            l_z,l_y,l_x = mask.shape[-3:]
            s_z,s_y,s_x = seg.shape[-3:]
            c_z,c_y,c_x = (min(l_z,s_z),min(l_y,s_y),min(l_x,s_x))

            seg = seg[:c_z,:c_y,:c_x] * mask[:c_z,:c_y,:c_x]

        else:
            seg = seg * mask

    if seg.shape != labels.shape:
        l_z,l_y,l_x = labels.shape[-3:]
        s_z,s_y,s_x = seg.shape[-3:]
        c_z,c_y,c_x = (min(l_z,s_z),min(l_y,s_y),min(l_x,s_x))

        labels = labels[:c_z,:c_y,:c_x]
        seg = seg[:c_z,:c_y,:c_x]

    print(f"num unique labels: {len(np.unique(labels))}")
    print(f"num unique seg: {len(np.unique(seg))}")

    #eval
    metrics = rand_voi(
        labels,
        seg,
        return_cluster_scores=True)

    metrics['merge_threshold'] = thresh
    metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']

    return metrics


if __name__ == "__main__":

    labels_file = sys.argv[1]
    labels_ds = sys.argv[2]
    seg_file = sys.argv[3]
    seg_ds = sys.argv[4]
    labels_mask = False#True

    #load labels, seg, labels_mask
    labels = open_ds(labels_file,labels_ds)
    seg = open_ds(seg_file,seg_ds)
    roi = seg.roi

    if labels_mask:
        mask = open_ds(labels_file,"labels_mask")
        mask = mask.to_ndarray(roi)
    else:
        mask = None

    #eval
    seg_arr = seg.to_ndarray(roi)
    labels_arr = labels.to_ndarray(roi)
    metrics = evaluate(seg_arr,labels_arr,mask)

    print(metrics)
