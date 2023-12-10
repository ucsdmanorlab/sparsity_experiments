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

def filter_fragments(fragments, affs, max_affinity_value=1.0,
        fragments_in_xy=True, filter_value=0.5):

    if affs.dtype == np.uint8:
        affs = (affs / np.max(affs)).astype(np.float32)

    if fragments_in_xy:
            average_affs = np.mean(affs[1:3]/max_affinity_value, axis=0)
    else:
        average_affs = np.mean(affs/max_affinity_value, axis=0)

    filtered_fragments = []

    fragment_ids = np.unique(fragments)

    for fragment, mean in zip(
            fragment_ids,
            measurements.mean(
                average_affs,
                fragments,
                fragment_ids)):

        if mean < filter_value:
            filtered_fragments.append(fragment)

    filtered_fragments = np.array(
        filtered_fragments,
        dtype=fragments.dtype)

    replace = np.zeros_like(filtered_fragments)

    replaced = replace_values(
            fragments,
            filtered_fragments,
            replace,
            inplace=False)

    return replaced


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    background_mask=False,
    mask_thresh=0.5,
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

            boundary_mask = mean_affs[z] > mask_thresh * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            if background_mask is False:
                boundary_mask = None

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
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

        boundary_mask = np.mean(affs, axis=0) > mask_thresh * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        if background_mask is False:
            boundary_mask = None

        ret = watershed_from_boundary_distance(
            boundary_distances, boundary_mask, return_seeds, min_seed_distance=min_seed_distance
        )

        fragments = ret[0]

    return ret


def watershed_from_boundary_distance(
    boundary_distances, boundary_mask, return_seeds=False, id_offset=0, min_seed_distance=10
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = mahotas.cwatershed(boundary_distances.max() - boundary_distances, seeds)

    if boundary_mask is not None:
        fragments *= boundary_mask

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def segment(
        pred_file,
        pred_dataset,
        roi=None,
        normalize_preds=False,
        min_seed_distance=10,
        background_mask=False,
        mask_thresh=0.5,
        filter_fragments_value=0.5,
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
    if normalize_preds:
        print("normalizing channel-wise")
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
   
    # watershed
    print("doing watershed")
    fragments = watershed_from_affinities(
        pred,
        fragments_in_xy=True,
        background_mask=background_mask,
        mask_thresh=mask_thresh,
        min_seed_distance=min_seed_distance)[0]
   
    # filter fragments
    if filter_fragments_value > 0:
        print("filtering fragments")
        fragments = filter_fragments(fragments, pred, filter_value=filter_fragments_value)

    fragment_ids = range(0,np.max(fragments))

    # get fragment centers
    fragment_centers = {
        fragment: roi.get_offset() + vs*Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments, fragments, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = nx.Graph()
    rag.add_nodes_from([
        (node, {
            'position_z': c[0],
            'position_y': c[1],
            'position_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])

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
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

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

    # find segments
    luts = {}
    if thresholds is None:
        thresholds = [round(i,2) for i in np.arange(0.25,0.85,0.05)]

    for thresh in thresholds:
        print(f"finding segments for {thresh}")
        luts[thresh] = find_connected_components(
                rag,
                node_component_attribute='segment_id',
                edge_score_attribute='merge_score',
                edge_score_threshold=thresh,
                return_lut=True)

    return luts, fragments, rag


if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    out_file = sys.argv[3]
    out_ds = sys.argv[4]
    rag_path = "SKEL/rag.db" #sys.argv[5]
    lut_file = f"SKEL/luts.json"
   
    raw_file = "/scratch/04101/vvenu/sparsity_experiments/voljo/data/train.zarr"
    #raw_file = "SKEL/test.zarr"
    labels_dataset = "labels"

    pred = open_ds(pred_file,pred_dataset)
    vs = pred.voxel_size
    roi = open_ds(raw_file,labels_dataset).roi
    roi = roi.intersect(pred.roi)

    try:
        thresh = float(sys.argv[5])
        merge_fn = sys.argv[6]
        norm = bool(int(sys.argv[7]))
        min_seed = int(sys.argv[8])
        bg_mask = bool(int(sys.argv[9]))
        mask_thresh = float(sys.argv[10])
        filter_val = float(sys.argv[11])
    except:
        thresh = 0.5
        merge_fn = "hist_quant_75"
        norm = False
        min_seed = 10
        bg_mask = False
        mask_thresh = 0.5
        filter_val = 0.0
    
    luts,frags,rag = segment(
            pred_file,
            pred_dataset,
            roi=[tuple(roi.offset),tuple(roi.shape)],
            normalize_preds=norm,
            min_seed_distance=min_seed,
            background_mask=bg_mask,
            mask_thresh=mask_thresh,
            filter_fragments_value=filter_val,
            merge_function=merge_fn,
            thresholds=None)
    
    out_seg = prepare_ds(
            out_file,
            out_ds,
            roi,
            vs,
            delete=True,
            dtype=np.uint64)
    
    out_frags = prepare_ds(
            out_file,
            "frags",
            roi,
            vs,
            delete=True,
            dtype=np.uint64)

    print("writing seg")
    out_frags[roi] = frags

    lut = np.vectorize(luts[thresh].get)
    seg = lut(frags)
    out_seg[roi] = seg.astype(np.uint64)
    
    print("writing rag")
    nx.write_graphml(rag, rag_path)
#    rag_provider = SQLiteGraphDataBase(
#            Path(rag_path),
#            position_attributes=["position_z", "position_y", "position_x"])
#    rag_provider.write_graph(rag, roi)

    print("writing luts")    
    def convert_dict(d):
        if isinstance(d, dict):
            return {int(k) if isinstance(k, np.uint64) else k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, np.uint64):
            return int(d)
        else:
            return d

    with open(lut_file, "w") as f:
        json.dump(convert_dict(luts),f,indent=4)
