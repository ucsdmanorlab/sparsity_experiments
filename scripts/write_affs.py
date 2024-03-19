import gunpowder as gp
from funlib.segment.graphs import find_connected_components
from funlib.segment.arrays import replace_values
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

import json
import logging
import sys
import numpy as np


def write_affs(f):

    neighborhood = [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]

    labels_ds = "labels_filtered_relabeled"
    out_ds = "gt_affs"
 
    labels = gp.ArrayKey('LABELS')
    affs = gp.ArrayKey('AFFS')

    # get rois
    roi = open_ds(f,labels_ds).roi
    voxel_size = open_ds(f,labels_ds).voxel_size

    input_shape = Coordinate((8,156,156))
    output_shape = Coordinate((4,128,128))

    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size
    context = (input_shape - output_shape) / 2

    total_input_roi = roi 
    total_output_roi = total_input_roi.grow(-context,-context).snap_to_grid(voxel_size,mode="shrink")

    chunk_request = gp.BatchRequest()
    chunk_request.add(labels, input_size)
    chunk_request.add(affs, output_size)

    prepare_ds(
        f,
        out_ds,
        total_output_roi,
        voxel_size,
        np.uint8,
        write_size=output_size,
        delete=True,
        num_channels=len(neighborhood)) 

    source = gp.ZarrSource(
                f,
                datasets = {
                    labels: labels_ds
                },
                array_specs = {
                    labels: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False)
                }
            )

    pipeline = source

    pipeline += gp.AddAffinities(
        neighborhood,
        labels,
        affs)
        
    pipeline += gp.IntensityScaleShift(affs,255,0)

    pipeline += gp.ZarrWrite(
            dataset_names={
                affs: out_ds
            },
            store=f
        )
    pipeline += gp.Scan(chunk_request)

    full_request = gp.BatchRequest()
    full_request[labels] = total_input_roi
    full_request[affs] = total_output_roi

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(full_request)
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    f = sys.argv[1]

    write_affs(f)
