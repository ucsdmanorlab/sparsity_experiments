import os
import sys
import numpy as np
import funlib.persistence
import kimimaro
import networkx as nx

def skeletonize(
        labels_file,
        labels_ds,
        affs_file_for_roi,
        teasar_params=None):

    if teasar_params is None:
        teasar_params={
            "scale": 1.5, 
            "const": 300, # physical units
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_acceptance_threshold": 3500, # physical units
            "soma_detection_threshold": 750, # physical units
            "soma_invalidation_const": 300, # physical units
            "soma_invalidation_scale": 2,
            "max_paths": 300, # default None
        }

    labels = funlib.persistence.open_ds(labels_file,labels_ds)
    roi = funlib.persistence.open_ds(affs_file_for_roi,"affs_50000").roi
    roi = roi.intersect(labels.roi)
    print(roi, labels.roi)
    labels_arr = labels.to_ndarray(roi=roi)
    vs = tuple(labels.voxel_size)

    skels = kimimaro.skeletonize(
            labels_arr,
            teasar_params,
            # object_ids=[ ... ], # process only the specified labels
            # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
            # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
            dust_threshold=100, # skip connected components with fewer than this many voxels
            anisotropy=vs, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            fill_holes=False, # default False
            fix_avocados=False, # default False
            progress=True, # default False, show progress bar
            parallel=10, # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=100, # how many skeletons to process before updating progress bar
    )

    uniques = np.unique(labels_arr)
    uniques = uniques[uniques > 0]
    skel_ids = np.array(list(skels.keys()))
    check = np.isin(uniques,skel_ids)

    if False not in check:
        return "good",skels
    else:
        print("missing skels!")
        print("s=",teasar_params["scale"])
        print("c=",teasar_params["const"])
        missing_ids = uniques[~check]
        #print(f"missing ids: {list(missing_ids)}")

        return f"bad_{len(missing_ids)}",skels, roi


def convert_to_nx(skels,roi):
    
    G = nx.Graph()
    node_offset = 0

    offset = roi.offset
        
    for skel in skels:
        
        skeleton = skels[skel]
        
        # Add nodes
        for vertex in skeleton.vertices:
            G.add_node(
                    node_offset, 
                    id=skeleton.id, 
                    position_z=vertex[0]+offset[0], 
                    position_y=vertex[1]+offset[1], 
                    position_x=vertex[2]+offset[2])

            node_offset += 1
    
        # Add edges
        for edge in skeleton.edges:
            adjusted_u = edge[0] + node_offset - len(skeleton.vertices)
            adjusted_v = edge[1] + node_offset - len(skeleton.vertices)
            G.add_edge(adjusted_u, adjusted_v, u=adjusted_u, v=adjusted_v)
        
    return G


if __name__ == '__main__':

    vols = ["cremi_a","cremi_b","cremi_c"]#,"epi","fib25","voljo"]
    fs = [f"/scratch/04101/vvenu/sparsity_experiments/{x}/data/train.zarr" for x in vols]

    for v in vols[::-1]:
        for s in [0,0.5,1.0,1.5,2.0,3.0,4.0]:
            for c in [0,100,200,300,400,500,750,1000]:

                f = f"/scratch/04101/vvenu/sparsity_experiments/{v}/data/train.zarr"
                affs_f = f"/scratch/04101/vvenu/sparsity_experiments/{v}/bootstrapped_nets/affs-2d_dense/rep_3/train.zarr"

                params={
                    "scale": s, 
                    "const": c, # physical units
                    "pdrf_scale": 100000,
                    "pdrf_exponent": 4,
                    "soma_acceptance_threshold": 3500, # physical units
                    "soma_detection_threshold": 750, # physical units
                    "soma_invalidation_const": 300, # physical units
                    "soma_invalidation_scale": 2,
                    "max_paths": 300, # default None
                }

                print("starting on ",f)
                print(f"scale={s}, const={c}")

                skels = skeletonize(f,"labels",affs_f,params)

                out_f = f"/scratch/04101/vvenu/sparsity_experiments/{v}/data/train_skels/n{len(skels[1])}_{skels[0]}_s{s}_c{c}.graphml"
                os.makedirs(os.path.dirname(out_f), exist_ok=True)
                
                G = convert_to_nx(skels[1],skels[2])
                print(f"writing..{out_f}")
                nx.write_graphml(G, out_f)
