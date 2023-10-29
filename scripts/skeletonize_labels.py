import sys
import numpy as np
import funlib.persistence
import kimimaro


def skeletonize(
        labels,
        roi,
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

    labels_arr = labels.to_ndarray(roi=roi)
    vs = tuple(labels.voxel_size)

    skels = kimimaro.skeletonize(
            labels_arr,
            teasar_params,
            # object_ids=[ ... ], # process only the specified labels
            # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
            # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
            dust_threshold=1000, # skip connected components with fewer than this many voxels
            anisotropy=vs, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True
            fill_holes=False, # default False
            fix_avocados=False, # default False
            progress=True, # default False, show progress bar
            parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=100, # how many skeletons to process before updating progress bar
    )

    return skels


if __name__ == '__main__':

    labels_file = "/scratch/04101/vvenu/sparsity_experiments/cremi_a/data/train.zarr"
    labels_dataset = "labels"
    roi = None

    labels = funlib.persistence.open_ds(labels_file, labels_dataset)

    skels = skeletonize(labels,roi)

    print(len(skels))
    for skel in skels.values():
        print(skel.id)
        print(skel.vertices)
        print(skel.edges)
        print(skel.radii)
        print(skel.vertex_types)
        print("")
