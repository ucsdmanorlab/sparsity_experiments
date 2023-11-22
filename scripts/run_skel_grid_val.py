import sys
import json
import os
import multiprocessing
import tqdm
import itertools
import gc

import numpy as np
import networkx as nx
from funlib.persistence import open_ds
from funlib.geometry import Coordinate,Roi
from funlib.evaluate import rand_voi, expected_run_length, split_graph
import hierarchical


def get_skeleton_lengths(
        skeletons,
        skeleton_position_attributes,
        skeleton_id_attribute,
        store_edge_length=None):
    '''Get the length of each skeleton in the given graph.

    Args:

        skeletons:

            A networkx-like graph.

        skeleton_position_attributes:

            A list of strings with the names of the node attributes for the
            spatial coordinates.

        skeleton_id_attribute:

            The name of the node attribute containing the skeleton ID.

        store_edge_length (optional):

            If given, stores the length of an edge in this edge attribute.
    '''

    node_positions = {
        node: np.array(
            [
                skeletons.nodes[node][d]
                for d in skeleton_position_attributes
            ],
            dtype=np.float32)
        for node in skeletons.nodes()
    }

    skeleton_lengths = {}
    for u, v, data in skeletons.edges(data=True):

        skeleton_id = skeletons.nodes[u][skeleton_id_attribute]

        if skeleton_id not in skeleton_lengths:
            skeleton_lengths[skeleton_id] = 0

        pos_u = node_positions[u]
        pos_v = node_positions[v]

        length = np.linalg.norm(pos_u - pos_v)

        if store_edge_length:
            data[store_edge_length] = length
        skeleton_lengths[skeleton_id] += length

    return skeleton_lengths


def get_site_fragment_lut(fragments, sites):
    '''Get the fragment IDs of all the sites.'''

    sites = list(sites)

    if len(sites) == 0:
        print(f"No sites in {roi}, skipping")
        return None, None

    fragment_ids = np.array([
        fragments[site['position_z'], site['position_y'], site['position_x']]
        for _,site in sites
    ])

    site_ids = np.array(
        [site['id'] for _,site in sites],
        dtype=np.uint64)

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]
    site_ids = site_ids[fg_mask]

    print(f"Got fragment IDs for {len(fragment_ids)} sites")

    lut = np.array([site_ids, fragment_ids])

    return lut, (fg_mask==0).sum()


def get_site_segment_ids(seg, sites, site_fragment_lut):

    sites = list(sites)
    
    if len(sites) == 0:
        print(f"No sites in {roi}, skipping")
        return None, None

    site_segment_ids = np.array([
        seg[site['position_z'], site['position_y'], site['position_x']]
        for _,site in sites
    ]).astype(np.uint64)

    fragment_segment_lut = np.array([site_fragment_lut[1],site_segment_ids]).astype(np.uint64)

#    fg_mask = seg_ids != 0
#    seg_ids = seg_ids[fg_mask]
#    site_ids = site_ids[fg_mask]

    return site_segment_ids, fragment_segment_lut


def compute_expected_run_length(skeletons, site_ids, site_segment_ids, skeleton_lengths):

    print("Calculating expected run length...")

    node_segment_lut = {
        site: segment for site, segment in zip(
            site_ids,
            site_segment_ids)
    }

    erl, stats = expected_run_length(
            skeletons=skeletons,
            skeleton_id_attribute='id',
            edge_length_attribute='length',
            node_segment_lut=node_segment_lut,
            skeleton_lengths=skeleton_lengths,
            return_merge_split_stats=True)

    perfect_lut = {
            node: data['id'] for node, data in \
                    skeletons.nodes(data=True)
    }

    max_erl, _ = expected_run_length(
            skeletons=skeletons,
            skeleton_id_attribute='id',
            edge_length_attribute='length',
            node_segment_lut=perfect_lut,
            skeleton_lengths=skeleton_lengths,
            return_merge_split_stats=True)

    split_stats = [
        {
            'comp_id': int(comp_id),
            'seg_ids': [(int(a), int(b)) for a, b in seg_ids]
        }
        for comp_id, seg_ids in stats['split_stats'].items()
    ]
    merge_stats = [
        {
            'seg_id': int(seg_id),
            'comp_ids': [int(comp_id) for comp_id in comp_ids]
        }
        for seg_id, comp_ids in stats['merge_stats'].items()
    ]

    return erl, max_erl, split_stats, merge_stats

def compute_splits_merges_needed(
        skeletons,
        rag_edges,
        site_ids,
        site_component_ids,
        site_fragment_ids,
        fragment_segment_lut,
        site_segment_ids,
        split_stats,
        merge_stats,
        threshold):

    total_splits_needed = 0
    total_additional_merges_needed = 0
    total_unsplittable_fragments = []

    print("Computing min-cut metric for each merging segment...")

    for i, merge in enumerate(merge_stats):

        print(f"Processing merge {i+1}/{len(merge_stats)}...")
        (
            splits_needed,
            additional_merges_needed,
            unsplittable_fragments) = mincut_metric(
                rag_edges,
                site_ids,
                site_component_ids,
                site_fragment_ids,
                fragment_segment_lut,
                site_segment_ids,
                merge['seg_id'],
                merge['comp_ids'],
                threshold)
        total_splits_needed += splits_needed
        total_additional_merges_needed += additional_merges_needed
        total_unsplittable_fragments += unsplittable_fragments

    total_merges_needed = 0
    for split in split_stats:
        total_merges_needed += len(split['seg_ids']) - 1
    total_merges_needed += total_additional_merges_needed

    return (
        total_splits_needed,
        total_merges_needed,
        total_unsplittable_fragments)

def mincut_metric(
        rag_edges,
        site_ids,
        site_component_ids,
        site_fragment_ids,
        fragment_segment_lut,
        site_segment_ids,
        segment_id,
        component_ids,
        threshold):

    # get RAG for segment ID
    rag = get_segment_rag(rag_edges, segment_id, fragment_segment_lut, threshold)

    print("Preparing RAG for split_graph call")

    # replace merge_score with weight
    for _, _, data in rag.edges(data=True):
        # print(_, _, data)
        data['weight'] = 1.0 - data['merge_score']

    # find fragments for each component in segment_id
    component_fragments = {}

    # True for every site that maps to segment_id
    segment_mask = site_segment_ids == segment_id

    # print('Component ids: ', component_ids)
    # print('Self site component ids: ', site_component_ids)

    for component_id in component_ids:

        # print('Component id: ', component_id)

        # limit following to sites that are part of component_id and
        # segment_id
        component_mask = site_component_ids == component_id
        fg_mask = site_fragment_ids != 0
        
        mask = np.logical_and(np.logical_and(component_mask, segment_mask), fg_mask)
        masked_site_ids = site_ids[mask]
        comp_site_fragment_ids = site_fragment_ids[mask]

        component_fragments[component_id] = comp_site_fragment_ids

        # print('Site ids: ', site_ids)
        # print('Site fragment ids: ', site_fragment_ids)

        for site_id, fragment_id in zip(masked_site_ids, comp_site_fragment_ids):

            if fragment_id == 0:
                continue

            # For each fragment containing a site, we need a position for
            # the split_graph call. We just take the position of the
            # skeleton node that maps to it, if there are several, we take
            # the last one.

            # print('Site id: ', site_id)
            # print('Fragment id: ', fragment_id, type(fragment_id))

            site_data = skeletons.nodes[site_id]
            fragment = rag.nodes[fragment_id]
            fragment['position_z'] = site_data['position_z']
            fragment['position_y'] = site_data['position_y']
            fragment['position_x'] = site_data['position_x']

            # Keep track of how many components share a fragment. If it is
            # more than one, this fragment is unsplittable.
            if 'component_ids' not in fragment:
                fragment['component_ids'] = set()
            fragment['component_ids'].add(component_id)

    # find all unsplittable fragments...
    unsplittable_fragments = []
    for fragment_id, data in rag.nodes(data=True):
        if fragment_id == 0:
            continue
        if 'component_ids' in data and len(data['component_ids']) > 1:
            unsplittable_fragments.append(fragment_id)
    # ...and remove them from the component lists
    for component_id in component_ids:

        fragment_ids = component_fragments[component_id]
        valid_mask = np.logical_not(
            np.isin(
                fragment_ids,
                unsplittable_fragments))
        valid_fragment_ids = fragment_ids[valid_mask]
        if len(valid_fragment_ids) > 0:
            component_fragments[component_id] = valid_fragment_ids
        else:
            del component_fragments[component_id]

    print(f"{len(unsplittable_fragments)} fragments are merging and can not be split")

    if len(component_fragments) <= 1:
        print(
            "after removing unsplittable fragments, there is nothing to "
            "do anymore")
        return 0, 0, unsplittable_fragments

    # these are the fragments that need to be split
    split_fragments = list(component_fragments.values())

    print("Splitting segment into {len(split_fragments)} components with sizes {[len(c) for c in split_fragments]}")

    print("Calling split_graph...")

    # call split_graph
    num_splits_needed = split_graph(
        rag,
        split_fragments,
        position_attributes=['position_z', 'position_y', 'position_x'],
        weight_attribute='weight',
        split_attribute='split')

    print(f"{num_splits_needed} splits needed for segment {segment_id}")

    # get number of additional merges needed after splitting the current
    # segment
    #
    # this is the number of split labels per component minus 1
    additional_merges_needed = 0
    for component, fragments in component_fragments.items():
        split_ids = np.unique([rag.node[f]['split'] for f in fragments])
        additional_merges_needed += len(split_ids) - 1

    print(f"{additional_merges_needed} additional merges needed to join components again")

    return (
        num_splits_needed,
        additional_merges_needed,
        unsplittable_fragments)

def get_segment_rag(rag_edges, segment_id, fragment_segment_lut, threshold):

    print("Reading RAG for segment ", segment_id)

    # get all fragments for the given segment
    segment_mask = fragment_segment_lut[1] == segment_id
    fragment_ids = fragment_segment_lut[0][segment_mask]

    # get the RAG containing all fragments
    nodes = [
        {'id': fragment_id, 'segment_id': segment_id}
        for fragment_id in fragment_ids
    ]

    rag = nx.Graph()
    node_list = [
        (n['id'], {'segment_id': n['segment_id']})
        for n in nodes
    ]

    edge_list = [
        (e['u'], e['v'], {'merge_score': e['score']})
        for e in rag_edges
        if e['score'] <= threshold
    ]

    print(f"RAG contains {len(node_list)} nodes/{len(edge_list)} edges")
    rag.add_nodes_from(node_list)
    rag.add_edges_from(edge_list)
    rag.remove_nodes_from([
        n
        for n, data in rag.nodes(data=True)
        if 'segment_id' not in data])

    print(
        "after filtering dangling node and not merged edges ",
        f"RAG contains {rag.number_of_nodes()} nodes/{rag.number_of_edges()} edges")

    return rag


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

    voxel_size = open_ds(raw_file,labels_dataset).voxel_size

    #run post
    segs, rags, frags = hierarchical.segment(
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
    skel_file = raw_file[:-5]+"_skel.graphml"
    skels = nx.read_graphml(skel_file)
    
    # convert positions to voxel space
    for node in skels.nodes:
        for i, attr in enumerate(['position_z', 'position_y', 'position_x']):
            skels.nodes[node][attr] = int((skels.nodes[node][attr] - roi.offset[i]) / voxel_size[i])

    # remove outside nodes and edges
    remove_nodes = []
    for node, data in skels.nodes(data=True):
        if 'position_z' not in data:
            remove_nodes.append(node)
        elif 'position_y' not in data:
            remove_nodes.append(node)
        elif 'position_x' not in data:
            remove_nodes.append(node)
        else:
            assert data['id'] >= 0

    print(f"Removing {len(remove_nodes)} nodes that were outside of ROI")

    for node in remove_nodes:
        skels.remove_node(node)

    skeletons = nx.Graph()

    # Add nodes with integer identifiers and their attributes
    for node, attrs in skels.nodes(data=True):
        skeletons.add_node(int(node), **attrs)

    # Add edges with updated node identifiers
    for u, v, attrs in skels.edges(data=True):
        skeletons.add_edge(int(u), int(v), **attrs)

    site_ids = np.array([
        n
        for n in skeletons.nodes()
    ], dtype=np.uint64)

    site_component_ids = np.array([
        data['id']
        for _, data in skeletons.nodes(data=True)
    ])

    site_component_ids = site_component_ids.astype(np.uint64)
    number_of_components = np.unique(site_component_ids).size

    skeleton_lengths = get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes=['position_z', 'position_y', 'position_x'],
            skeleton_id_attribute='id',
            store_edge_length='length')
    total_length = np.sum([l for _, l in skeleton_lengths.items()])

    # prepare fragments
    site_fragment_lut, num_bg_sites = get_site_fragment_lut(frags,skeletons.nodes(data=True))

    _site_fragment_lut = {
        site: fragment
        for site, fragment in zip(
            site_fragment_lut[0],
            site_fragment_lut[1])
    }

    site_fragment_ids = np.array([
        _site_fragment_lut[s] if s in _site_fragment_lut else 0
        for s in site_ids
    ], dtype=np.uint64)

    #evaluate thresholds
    results = {}
    
    for thresh, seg in segs.items():
        
        site_segment_ids, fragment_segment_lut = get_site_segment_ids(
                seg,
                skeletons.nodes(data=True),
                site_fragment_lut
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
                rags[thresh],
                site_ids,
                site_component_ids,
                site_fragment_ids,
                fragment_segment_lut,
                site_segment_ids,
                split_stats,
                merge_stats,
                thresh)

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
        metrics['number_of_background_sites'] = int(num_bg_sites)

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
        n_workers = 1

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
