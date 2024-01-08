import sys
import json
import os
import multiprocessing as mp
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


def convert_dtypes(d):
    if isinstance(d, dict):
        return {k: convert_dtypes(v) for k, v in d.items()}
    elif isinstance(d, (np.int64, np.uint64, np.int32, np.uint32)):
        return int(d)
    elif isinstance(d, (np.float64, np.float32, np.float16)):
        return float(d)
    return d


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

    roi_offset = roi.get_offset()
    roi_shape = roi.get_shape()
    compute_mincut_metric = True 

    frags_file = pred_file 
    frags_str = f"{pred_dataset}_{normalize_preds}Norm_{background_mask}BoundaryMask{int(100*mask_thresh)}_{min_seed_distance}MinSeedDist_{int(100*filter_fragments_value)}FragFilter"
    frags_ds = f"repost/{frags_str}/fragments"
    edges_table = "edges_"+merge_function
    rag_path = os.path.join(frags_file,"repost",frags_str,"rag.db")
    lut_dir = os.path.join(os.path.dirname(rag_path),"luts",merge_function)

    # evaluate
    evaluate = EvaluateAnnotations(
            frags_file,
            frags_ds,
            rag_path,
            edges_table,
            lut_dir,
            roi_offset,
            roi_shape,
            compute_mincut_metric)
    
    results = evaluate.evaluate()
    results = convert_dtypes(results)

    #get best result
    best_nvi_thresh = sorted([(results[thresh]['nvi_sum'],thresh) for thresh in results.keys()])
    best_edits_thresh = sorted([
        (results[thresh]['total_splits_needed_to_fix_merges'] + results[thresh]['total_merges_needed_to_fix_splits'],thresh) 
        for thresh in results.keys()
    ])
    
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

    for arg_tuple in missing_arg_tuples:
        i,result = eval_run(arg_tuple)
        print(arg_tuple, i, result["best_nvi"]["nvi_sum"])
        with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
            json.dump(result,f,indent=4)

#    with mp.get_context('spawn').Pool(n_workers,maxtasksperchild=1) as pool:
#
#        for i,result in tqdm.tqdm(pool.imap(eval_run,missing_arg_tuples), total=len(missing_arg_tuples)):
#
#            with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
#                json.dump(result,f,indent=4)
