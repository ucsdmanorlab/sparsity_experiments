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
import mws


def evaluate(
    seg,
    labels,
    mask=None):

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

    metrics['voi_sum'] = metrics['voi_split']+metrics['voi_merge']
    metrics['nvi_sum'] = metrics['nvi_split']+metrics['nvi_merge']

    for k in {'voi_split_i', 'voi_merge_j'}:
        del metrics[k]

    return metrics


def eval_run(args):

    roi = args["roi"]

    raw_file = args["raw_file"]
    labels_dataset = args["labels_dataset"] #+ f"/{roi}"
    labels_mask = args["labels_mask"] #+ f"/{roi}"
    pred_file = args["pred_file"]
    pred_dataset = args["pred_dataset"] #+ f"/{roi}"
    normalize_preds = args["normalize_preds"]
    stride = args["stride"]
    randomize_strides = args["randomize_strides"]
    algorithm = args["algorithm"]
    neighborhood = args["neighborhood"]
    mask_thresh = args["mask_thresh"]
    erode_steps = args["erode_steps"]
    clean_up = args["clean_up"]

    #run post
    seg = mws.post(
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
        clean_up)

    #get roi
    if roi is not None:
        pred = open_ds(pred_file,pred_dataset)
        roi = Roi(pred.roi.offset+Coordinate(roi[0]),roi[1])
    else:
        roi = open_ds(pred_file,pred_dataset).roi

    # load labels, labels_mask
    labels = open_ds(raw_file,labels_dataset).to_ndarray(roi,fill_value=0)
    
    if labels_mask:
        mask = open_ds(raw_file,labels_mask)
        mask = mask.to_ndarray(roi,fill_value=0)
    else:
        mask = None
        
    # evaluate
    results = evaluate(seg,labels,mask)

    return args | results


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

    #with multiprocessing.Pool(n_workers,maxtasksperchild=1) as pool:
    with multiprocessing.get_context('spawn').Pool(n_workers,maxtasksperchild=1) as pool:

        for i,result in enumerate(tqdm.tqdm(pool.imap(eval_run,arguments), total=len(arguments))):

            with open(os.path.join(out_dir,f"{i}.json"),"w") as f:
                json.dump(result,f,indent=4)
