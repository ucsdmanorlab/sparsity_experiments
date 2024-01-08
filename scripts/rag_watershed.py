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
from funlib.segment.arrays import replace_values

import mahotas
import waterz


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


def watershed(
        pred_file,
        pred_dataset,
        fragments_file,
        roi=None,
        normalize_preds=False,
        min_seed_distance=10,
        background_mask=False,
        mask_thresh=0.5,
        filter_fragments_value=0.5,
        epsilon_agglomerate=0.05,
        **kwargs):

    frag_str = f"{pred_dataset}_{normalize_preds}Norm_{background_mask}BoundaryMask{int(100*mask_thresh)}_{min_seed_distance}MinSeedDist_{int(100*filter_fragments_value)}FragFilter"
    fragments_dataset = os.path.join("repost",frag_str,"fragments")
    rag_path = os.path.join(fragments_file,"repost",frag_str,"rag.db")

    # load
    print("loading affs")
    pred = open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = Roi(roi[0],roi[1])
    else:
        roi = pred.roi

    vs = pred.voxel_size

    # to store nodes and fragments
    out_frags = prepare_ds(
            fragments_file,
            fragments_dataset,
            roi,
            vs,
            delete=True,
            compressor=dict(id='blosc'),
            dtype=np.uint64)

    rag_provider = SQLiteGraphDataBase(
            Path(rag_path),
            mode="w",
            position_attributes=["position_z", "position_y", "position_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool})
    rag = rag_provider[roi]

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

    # epsilon agglomerate
    if epsilon_agglomerate > 0:
        generator = waterz.agglomerate(
                affs=pred,
                thresholds=[epsilon_agglomerate],
                fragments=fragments,
                scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                discretize_queue=256,
                return_merge_history=False,
                return_region_graph=False)
        fragments[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass 

    fragment_ids = range(0,np.max(fragments))

    # get fragment centers
    print("getting fragment centers")
    fragment_centers = {
        fragment: roi.get_offset() + vs*Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments, fragments, fragment_ids))
        if not np.isnan(center[0])
    }

    # write
    rag.add_nodes_from([
        (node, {
            'position_z': c[0],
            'position_y': c[1],
            'position_x': c[2]
            }
        )
        for node, c in fragment_centers.items()
    ])
    
    print("writing frags")
    out_frags[roi] = fragments

    print("writing rag")
    rag_provider.write_graph(rag, roi)

    return fragments_dataset, rag_path


if __name__ == "__main__":

    pred_file = sys.argv[1]
    pred_dataset = sys.argv[2]
    out_file = sys.argv[3]
   
    pred = open_ds(pred_file,pred_dataset)
    vs = pred.voxel_size
    roi = pred.roi

    try:
        norm = bool(int(sys.argv[4]))
        bg_mask = bool(int(sys.argv[5]))
        mask_thresh = float(sys.argv[6])
        min_seed = int(sys.argv[7])
        filter_val = float(sys.argv[8])
    except:
        norm = False
        min_seed = 15
        bg_mask = False
        mask_thresh = 0.4
        filter_val = 0.0
    
    frag_ds, rag_path = watershed(
            pred_file,
            pred_dataset,
            out_file,
            roi=[tuple(roi.offset),tuple(roi.shape)],
            normalize_preds=norm,
            min_seed_distance=min_seed,
            background_mask=bg_mask,
            mask_thresh=mask_thresh,
            filter_fragments_value=filter_val)

    print(frag_ds, rag_path)
