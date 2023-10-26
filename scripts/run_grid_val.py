import sys
import json
import os
import multiprocessing
import tqdm
import itertools
import gc

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

    #eval
    metrics = rand_voi(
        labels,
        seg,
        return_cluster_scores=True)

    metrics['merge_threshold'] = thresh
    metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']

    return metrics


def eval_run(arg_tuple):

    idx,args = arg_tuple

    roi = args["roi"]

    raw_file = args["raw_file"]
    labels_dataset = args["labels_dataset"] #+ f"/{roi}"
    labels_mask = args["labels_mask"] #+ f"/{roi}"
    pred_file = args["pred_file"]
    pred_dataset = args["pred_dataset"] #+ f"/{roi}"
    normalize_preds = args["normalize_preds"]
    min_seed_distance = args["min_seed_distance"]
    merge_function = args["merge_function"]
    erode_steps = args["erode_steps"]
    clean_up = args["clean_up"]

    if 'test_50000' in pred_file:
        pred_dataset = 'affs_20000'

    #get roi
    if roi is not None:
        pred = open_ds(pred_file,pred_dataset)
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        pred_roi = open_ds(pred_file,pred_dataset).roi
        roi = open_ds(raw_file,labels_dataset).roi
        roi = roi.intersect(pred_roi)

    #run post
    segs,frags = hierarchical.post(
            pred_file,
            pred_dataset,
            roi=[tuple(roi.offset - pred_roi.offset),tuple(roi.shape)],
            normalize_preds=normalize_preds,
            min_seed_distance=min_seed_distance,
            merge_function=merge_function,
            thresholds=None,
            erode_steps=erode_steps,
            clean_up=clean_up)

    #load labels, labels_mask
    labels = open_ds(raw_file,labels_dataset).to_ndarray(roi,fill_value=0)
    
    if labels_mask:
        mask = open_ds(raw_file,labels_mask)
        mask = mask.to_ndarray(roi, fill_value=0)
    else:
        mask = None

    #eval frags
    frags_metrics = evaluate(frags,labels,mask)

    del frags
    gc.collect()

    #eval segs
    results = {}
    
    for thresh, seg in segs.items():

        metrics = evaluate(seg,labels,mask,thresh)
        results[thresh] = metrics

    del segs
    gc.collect()

    #get best result
    best_thresh = sorted([(results[thresh]['nvi_sum'],thresh) for thresh in results.keys()])
    
    try:
        best_thresh = best_thresh[0][1]
    except:
        print(results.keys())
        print(results)
        print(best_thresh)
        print(segs.keys())
        print(args)

    ret = args | {"frags": frags_metrics} | {"best": results[best_thresh]}
    return idx, ret


if __name__ == "__main__":

    jsonfile = sys.argv[1]
    try:
        n_workers = int(sys.argv[2])
    except:
        n_workers = 10

    out_dir = jsonfile.split(".")[0]
    os.makedirs(out_dir,exist_ok=True)

    #load args
    with open(jsonfile,"r") as f:
        arguments = json.load(f)

    keys, values = zip(*arguments.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    #get existing results
    existing_outputs = [int(f.split(".")[0]) for f in os.listdir(out_dir) if f.endswith('.json')]

    #get missing results
    total_indices = list(range(len(arguments)))
    missing_indices = list(set(total_indices) - set(existing_outputs))
    missing_arg_tuples = [(idx, arguments[idx]) for idx in missing_indices]

    #with multiprocessing.Pool(n_workers,maxtasksperchild=1) as pool:
    with multiprocessing.get_context('spawn').Pool(n_workers,maxtasksperchild=1) as pool:

        for i,result in tqdm.tqdm(pool.imap(eval_run,missing_arg_tuples), total=len(missing_arg_tuples)):

            with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
                json.dump(result,f,indent=4)
