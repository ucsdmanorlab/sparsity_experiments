import sys
import os
import numpy as np
import zarr
import json
import multiprocessing
import tqdm
import itertools

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate,Roi
from funlib.segment.arrays import replace_values, relabel
from funlib.evaluate import rand_voi

import scipy.ndimage
from scipy.ndimage import  gaussian_filter

import mwatershed as mws


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


def evaluate(
    seg,
    labels,
    mask=None,
    filter_amt=None):
    
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

    #eval
    metrics = rand_voi(
        labels,
        seg,
        return_cluster_scores=False)

    metrics['filter_fragments'] = filter_amt
    metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]
        
    return metrics


def segment_eval(
        raw_file,
        labels_dataset,
        pred_file,
        pred_dataset,
        roi=None,
        normalize_preds=False,
        seeds_file=None,
        seeds_dataset=None,
        labels_mask=False,
        **kwargs):

    # load pred
    pred = open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    # load labels, labels_mask
    labels = open_ds(raw_file,labels_dataset).to_ndarray(roi)
    
    if labels_mask:
        mask = open_ds(raw_file,labels_mask)
        mask = mask.to_ndarray(roi)
    else:
        mask = None
        
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
  
    # average affs
    average_affs: float = np.mean(pred, axis=0)

    # normalize channel-wise
    if normalize_preds:
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])

    # add noise and smooth
    pred = pred + random_noise + smoothed_affs

    # load seeds
    if seeds_file is not None and seeds_dataset is not None:
        seeds = open_ds(seeds_file,seeds_dataset).to_ndarray(roi)
    else:
        seeds = None

    # store results here
    results = []

    # looping params
    for lr_bias in [0.0,-0.1,-0.125, -0.15,-0.175,-0.2,-0.225, -0.25, -0.3]:
        for adj_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

            # bias towards splitting
            shift: np.ndarray = np.array(
                [
                    adj_bias if abs(min(offset)) <= 1
                    else np.linalg.norm(offset) * lr_bias
                    for offset in offsets
                ]
            ).reshape((-1, *((1,) * (len(pred.shape) - 1))))

            # do mws
            seg = mws.agglom(
                    pred + shift,
                    offsets,
                    seeds)

            # filter fragments
            fragment_ids: np.ndarray = np.unique(seg)
            mean_aff_values = scipy.ndimage.mean(average_affs, seg, fragment_ids)

            for filter_val in [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6]:

                filtered_seg = seg.copy()

                if filter_val > 0.0:
                    filtered_fragments: list = []

                    for fragment, mean in zip(fragment_ids, mean_aff_values):
                        if mean < filter_val:
                            filtered_fragments.append(fragment)

                    filtered_fragments: np.ndarray = np.array(filtered_fragments, dtype=seg.dtype)
                    replace: np.ndarray = np.zeros_like(filtered_fragments)
                    replace_values(filtered_seg, filtered_fragments, replace, inplace=True)

                # evaluate
                result = evaluate(filtered_seg,labels,mask,filter_val)
                result['adj_bias'] = adj_bias
                result['lr_bias'] = lr_bias

                results.append(result)

    # return best
    results = sorted(results, key=lambda x: x["nvi_sum"])
    print(len(results))

    return results[0]


def eval_run(args):

#    result = segment_eval(
#        raw_file,
#        labels_dataset,
#        pred_file,
#        pred_dataset,
#        roi=None,
#        normalize_preds=False,
#        seeds_file=None,
#        seeds_dataset=None,
#        labels_mask=False)
    result = segment_eval(**args)

    print(args | result)
    return args | result


if __name__ == "__main__":

    jsonfile = sys.argv[1]
    n_workers = int(sys.argv[2])

    out_dir = jsonfile.split(".")[0]
    os.makedirs(out_dir,exist_ok=True)

    #load args
    with open(jsonfile,"r") as f:
        arguments = json.load(f)

    keys, values = zip(*arguments.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    with multiprocessing.Pool(n_workers,maxtasksperchild=1) as pool:
    #with multiprocessing.get_context('spawn').Pool(n_workers,maxtasksperchild=1) as pool:

        for i,result in enumerate(tqdm.tqdm(pool.imap(eval_run,arguments), total=len(arguments))):

            with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
                json.dump(result,f,indent=4)
