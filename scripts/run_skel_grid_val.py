import sys
import json
import os
import multiprocessing as mp
import concurrent.futures
import tqdm
import itertools
import gc

import numpy as np
import networkx as nx
from funlib.persistence import open_ds
from funlib.geometry import Coordinate,Roi
from funlib.evaluate import rand_voi, expected_run_length, split_graph
from rag_hierarchical import segment
from evaluate import EvaluateAnnotations, get_site_fragment_lut


def eval_run(arg_tuple):

    idx,args = arg_tuple

    roi = args["roi"]

    raw_file = args["raw_file"]
    labels_dataset = args["labels_dataset"] #+ f"/{roi}"
    labels_mask = args["labels_mask"] #+ f"/{roi}"
    pred_file = args["pred_file"]
    pred_dataset = args["pred_dataset"] #+ f"/{roi}"
    normalize_preds = args["normalize_preds"]
    background_mask = args["background_mask"]
    mask_thresh = args["mask_thresh"]
    min_seed_distance = args["min_seed_distance"]
    filter_fragments_value = args["filter_fragments_value"]
    merge_function = args["merge_function"]
    rso = args["rso"]

    if 'test_50000' in pred_file:
        pred_dataset = 'affs_20000'

    #get roi
    if roi is not None:
        roi = Roi(roi[0],roi[1])
    else:
        pred_roi = open_ds(pred_file,pred_dataset).roi
        roi = open_ds(raw_file,labels_dataset).roi
        roi = roi.intersect(pred_roi)

    voxel_size = open_ds(raw_file,labels_dataset).voxel_size

    #run post
    luts, frags, rag = segment(
            pred_file,
            pred_dataset,
            roi=[tuple(roi.offset),tuple(roi.shape)],
            normalize_preds=normalize_preds,
            min_seed_distance=min_seed_distance,
            background_mask=background_mask,
            mask_thresh=mask_thresh,
            filter_fragments_value=filter_fragments_value,
            merge_function=merge_function,
            thresholds=None)

    roi_offset = roi.get_offset()
    roi_shape = roi.get_shape()
    compute_mincut_metric = True 
    skel_file = raw_file[:-5]+"_skel.graphml"

    #evaluate
    evaluate = EvaluateAnnotations(
            raw_file,
            labels_dataset,
            luts,
            frags,
            rag,
            skel_file,
            roi_offset,
            roi_shape,
            compute_mincut_metric)
    
    results = evaluate.evaluate()

    #get best result
    best_nvi_thresh = sorted([(results[thresh]['nvi_sum'],thresh) for thresh in results.keys()])
    best_edits_thresh = sorted([
        (results[thresh]['total_splits_needed_to_fix_merges'] + results[thresh]['total_merges_needed_to_fix_splits'],thresh) 
        for thresh in results.keys()
    ])
    
    try:
        best_nvi_thresh = best_nvi_thresh[0][1]
        best_edits_thresh = best_edits_thresh[0][1]
    except:
        print(results)
        print(args)

    ret = args | {"best_nvi": results[best_nvi_thresh]} | {"best_edits": results[best_edits_thresh]}
    return idx, ret


if __name__ == "__main__":

    jsonfile = sys.argv[1]
    part = int(sys.argv[2])
    try:
        n_workers = int(sys.argv[3])
    except:
        n_workers = 1

    parts = 64

    out_dir = jsonfile.split(".")[0]+f"_{part}"
    os.makedirs(out_dir,exist_ok=True)

    #load args
    with open(jsonfile,"r") as f:
        arguments = json.load(f)

    keys, values = zip(*arguments.items())
    arguments = [dict(zip(keys, v)) for v in itertools.product(*values)][part::parts]

    #get existing results
    existing_outputs = [int(f.split(".")[0]) for f in os.listdir(out_dir) if f.endswith('.json')]

    #get missing results
    total_indices = list(range(len(arguments)))
    missing_indices = list(set(total_indices) - set(existing_outputs))
    missing_arg_tuples = [(idx, arguments[idx]) for idx in missing_indices]

    with mp.get_context('spawn').Pool(n_workers,maxtasksperchild=1) as pool:

        for i,result in tqdm.tqdm(pool.imap(eval_run,missing_arg_tuples), total=len(missing_arg_tuples)):

            with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
                json.dump(result,f,indent=4)
