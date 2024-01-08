import json
import sys
import os
import numpy as np
from pathlib import Path
import networkx as nx

from scipy.ndimage import measurements, center_of_mass
from scipy.ndimage import maximum_filter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion
from skimage.measure import label

import zarr
from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.graphs import SQLiteGraphDataBase
from funlib.geometry import Coordinate,Roi
from funlib.segment.arrays import replace_values, relabel
from funlib.segment.graphs import find_connected_components

import mahotas
import waterz
from lsd.post.merge_tree import MergeTree


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

def agglomerate(
        pred_file,
        pred_dataset,
        fragments_file,
        fragments_dataset,
        rag_path,
        roi=None,
        merge_function="mean",
        thresholds=None,
        **kwargs):

    # load
    print("loading affs")
    pred = open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = Roi(roi[0],roi[1])
    else:
        roi = pred.roi

    vs = pred.voxel_size

    # first three channels are direct neighbor affs
    pred = pred.to_ndarray(roi)[:3]
    
    # normalize
    print("converting to float")
    pred = (pred / np.max(pred)).astype(np.float32)
    
    # normalize channel-wise
    normalize_preds = "True" in fragments_dataset.split("_")[2] 
    if normalize_preds:
        print("normalizing channel-wise")
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
   
    # load fragments
    print("loading frags")
    fragments = open_ds(fragments_file,fragments_dataset).to_ndarray(roi)
  
    # load rag
    print("loading rag nodes")
    rag_provider = SQLiteGraphDataBase(
            Path(rag_path),
            mode="r+",
            position_attributes=["position_z", "position_y", "position_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool},
            edges_table="edges_"+merge_function)
    rag = rag_provider.read_graph(roi=roi)

    # relabel fragments
    fragments_relabelled, n, fragment_relabel_map = relabel(
        fragments,
        return_backwards_map=True)

    # agglomerate
    print("agglomerating")
    generator = waterz.agglomerate(
            pred,
            thresholds=[0,1.0],
            fragments=fragments_relabelled.copy(),
            scoring_function=waterz_merge_function[merge_function],
            return_merge_history=True,
            return_region_graph=True,
            )

    # get fragment neighbors
    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge['u']], fragment_relabel_map[edge['v']]
        rag.add_edge(int(u), int(v), merge_score=None, agglomerated=True)

    #_, merge_rag = next(generator)
    _, merge_history, _ = next(generator)

    # cleanup generator
    for _, _, _ in generator:
        pass

    # create a merge tree from the merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:

        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score)

    # mark edges in original RAG with score at time of merging
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data['merge_score'] = merge_score

    # update rag
    print("writing rag edges")
    rag_provider.write_edges(rag.nodes,rag.edges,roi)

    # find segments
    if thresholds is None:
        thresholds = [float(round(i,2)) for i in np.arange(0.2,0.9,0.05)]

    print("writing luts")
    lut_dir = os.path.join(os.path.dirname(rag_path),"luts",merge_function)
    os.makedirs(lut_dir, exist_ok=True)
    for thresh in thresholds:
        lut_name = f"thresh_{int(thresh*100)}"
        lut = find_connected_components(
                rag,
                #node_component_attribute='segment_id',
                edge_score_attribute='merge_score',
                edge_score_threshold=thresh,
                return_lut=True)
        
        lut = np.stack([list(lut.keys()),list(lut.values())]).astype(np.uint64)
        print(f"writing {os.path.join(lut_dir,lut_name)}.npy")
        np.save(os.path.join(lut_dir,lut_name), lut)


if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    frags_file = sys.argv[3]
    frags_dataset = sys.argv[4]
    rag_path = sys.argv[5]
    merge_fn = sys.argv[6]

    pred = open_ds(pred_file,pred_dataset)
    frags = open_ds(frags_file,frags_dataset)
    vs = pred.voxel_size
    roi = frags.roi
    
    agglomerate(
        pred_file,
        pred_dataset,
        frags_file,
        frags_dataset,
        rag_path,
        roi=[tuple(roi.offset),tuple(roi.shape)],
        merge_function=merge_fn,
        thresholds=None)
