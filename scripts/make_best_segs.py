import sys
import json

import numpy as np
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds
import multiprocessing as mp

from hierarchical import post


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

default = {
    "pred_file": "",
    "pred_dataset": "",
    "roi":None,
    "normalize_preds":0,
    "min_seed_distance":10,
    "merge_function":"mean",
    "thresholds":None,
    "erode_steps":0,
    "clean_up":0
}

def make_seg(arg):
        print("making", arg)

        segs,_ = post(**arg)
        
        pred = open_ds(arg["pred_file"],arg["pred_dataset"])

        if arg["roi"] is not None:
            roi = Roi(pred.roi.offset+Coordinate(arg["roi"][0]),arg["roi"][1])
        else:
            roi = pred.roi
        
        print("writing")

        out_seg = prepare_ds(
                arg["out_file"],
                arg["out_ds"],
                roi,
                pred.voxel_size,
                np.uint64)

        out_seg[roi] = segs[float(arg["thresholds"][0])]


if __name__ == "__main__":

    best_segs_json = sys.argv[1]

    with open(best_segs_json,"r") as f:
        b = json.load(f)

    print("making arguments")
    #get all arguments
    arguments = []
    for d in ["fib25"]:
        for n in b[d]:
            for s in b[d][n]:
                for r in b[d][n][s]:
                    
                    res = b[d][n][s][r]
                    p_i = res["pred_iteration"]
                    a_i = res["affs_iteration"]
                    n_ = n.split('_')[-1] #sparse net
                    n_ = "lsd" if n_ == "lsds" else n_
                    p = n.split('_')[0] #pred
                    
                    arg = default.copy()
                    
                    if s != '3d_dense':
                        arg["pred_file"] = f"/scratch/04101/vvenu/sparsity_experiments/{d}/affs_nets/{p}/{res['gt_type']}/2d_test.zarr"
                        arg["pred_dataset"] = f"{n_}/{s}/{r}/3d_affs_{a_i}_from_stacked_{p}_{p_i}"

                    elif s == '3d_dense' and n.startswith('lsd'):
                        arg["pred_file"] = f"/scratch/04101/vvenu/sparsity_experiments/{d}/affs_nets/dense_3d_lsds/{res['gt_type']}/{n_}/{r}/test_{p_i}.zarr"
                        arg["pred_dataset"] = f"affs_{a_i}"
                    
                    else:
                        arg["pred_file"] = f"/scratch/04101/vvenu/sparsity_experiments/{d}/sparse_nets/{n_}/{s}/{r}/test.zarr"
                        arg["pred_dataset"] = f"affs_{p_i}"


                    arg["out_file"] = f"/scratch/04101/vvenu/sparsity_experiments/{d}/data/test.zarr"
                    arg["out_ds"] = f"{n}/{s}/{r}/seg"
                    arg["normalize_preds"] = bool(res["normalize_preds"])
                    arg["merge_function"] = res["merge_function"]
                    arg["thresholds"] = [res["merge_threshold"]]

                    arguments.append(arg)
    
#    for arg in arguments:
#        make_seg(arg)

    with mp.get_context('spawn').Pool(16,maxtasksperchild=1) as pool:
        pool.map(make_seg,arguments[::-1])
