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


def compute_rand_voi(
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

    #run post
    segs, frags = hierarchical.segment(
            pred_file,
            pred_dataset,
            roi=[tuple(roi.offset),tuple(roi.shape)],
            normalize_preds=normalize_preds,
            min_seed_distance=min_seed_distance,
            background_mask=background_mask,
            mask_thresh=mask_thresh,
            filter_fragments_value=filter_fragments_value,
            merge_function=merge_function,
            thresholds=None,
            rso=rso)

    #load labels, labels_mask
    labels = open_ds(raw_file,labels_dataset).to_ndarray(roi,fill_value=0)
    
    if labels_mask:
        mask = open_ds(raw_file,labels_mask)
        mask = mask.to_ndarray(roi, fill_value=0)
    else:
        mask = None

    # get gt skeletons
    skeletons = get_skeletons(labels,roi)

    # TODO: remove outside nodes and edges

    site_ids = np.array([
        n
        for n in skeletons.nodes()
    ], dtype=np.uint64)

    site_component_ids = np.array([
        data['component_id']
        for _, data in skeletons.nodes(data=True)
    ])

    site_component_ids = site_component_ids.astype(np.uint64)
    number_of_components = np.unique(site_component_ids).size

    skeleton_lengths = get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes=['z', 'y', 'x'],
            skeleton_id_attribute='component_id',
            store_edge_length='length')
    total_length = np.sum([l for _, l in skeleton_lengths.items()])

    # prepare fragments
    site_fragment_lut, num_bg_sites = get_site_fragment_lut(fragments,site_ids)

    site_fragment_lut = {
        site: fragment
        for site, fragment in zip(
            site_fragment_lut[0],
            site_fragment_lut[1])
    }

    site_fragment_ids = np.array([
        site_fragment_lut[s] if s in site_fragment_lut else 0
        for s in site_ids
    ], dtype=np.uint64)

    #evaluate thresholds
    results = {}
    
    for thresh, seg in segs.items():
        
        site_segment_ids, fragment_segment_lut = get_site_segment_ids(
                seg,
                site_ids,
                site_fragment_ids
        )
        
        number_of_segments = np.unique(site_segment_ids).size

        # compute ERL, get split-merge stats
        erl, max_erl, split_stats, merge_stats = compute_expected_run_length(
                skeletons,
                site_ids,
                site_segment_ids,
                skeleton_lengths)

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        print('ERL: ', erl)
        print('Max ERL: ', max_erl)
        print('Total path length: ', total_length)

        normalized_erl = erl/max_erl
        print('Normalized ERL: ', normalized_erl)

        # compute mincut metric
        splits_needed, merges_needed, unsplittable_fragments = \
            compute_splits_merges_needed(
                skeletons,
                site_ids,
                site_component_ids,
                site_fragment_ids,
                fragment_segment_lut,
                site_segment_ids,
                split_stats,
                merge_stats,
                threshold)

        average_splits_needed = splits_needed/number_of_segments
        average_merges_needed = merges_needed/number_of_components
        print(
                'Number of splits needed: ', splits_needed, '\n',
                'Number of merges needed: ', merges_needed, '\n',
                'Number of background sites: ', num_bg_sites, '\n',
                'Average splits needed: ', average_splits_needed, '\n',
                'Average merges needed: ', average_merges_needed, '\n',
                'Number of unsplittable fragments: ', len(unsplittable_fragments)
            )

        # compute RAND VOI
        metrics = compute_rand_voi(seg,labels,mask,thresh)

        metrics['expected_run_length'] = erl
        metrics['max_erl'] = max_erl
        metrics['total path length'] = total_length
        metrics['normalized_erl'] = normalized_erl
        metrics['number_of_segments'] = number_of_segments
        metrics['number_of_components'] = number_of_components
        metrics['number_of_merging_segments'] = number_of_merging_segments
        metrics['number_of_split_skeletons'] = number_of_split_skeletons

        metrics['total_splits_needed_to_fix_merges'] = splits_needed
        metrics['average_splits_needed_to_fix_merges'] = average_splits_needed
        metrics['total_merges_needed_to_fix_splits'] = merges_needed
        metrics['average_merges_needed_to_fix_splits'] = average_merges_needed
        metrics['number_of_unsplittable_fragments'] = len(unsplittable_fragments)
        metrics['number_of_background_sites'] = num_bg_sites

        metrics['merge_stats'] = merge_stats
        metrics['split_stats'] = split_stats

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

    ret = args | {"best": results[best_thresh]}
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
