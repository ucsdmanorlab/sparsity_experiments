import json
import re
import shutil
import zarr
import glob
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import daisy
from funlib.persistence import prepare_ds

from model import MtlsdModel


setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

out_shapes = config["output_shapes"]

increase = gp.Coordinate([16, 8*12, 8*12])
input_shape = gp.Coordinate(tuple(config['input_shape'])) + increase
output_shape = gp.Coordinate(tuple(config['output_shape'])) + increase

# nm
voxel_size = gp.Coordinate(tuple(config['voxel_size']))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size
context = (input_size - output_size) / 2


def predict(
        iteration,
        raw_file,
        raw_dataset,
        out_file):

    lsds_out_ds = f"lsds_{iteration}"
    affs_out_ds = f"affs_{iteration}"

    raw = gp.ArrayKey('RAW')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)
    scan_request.add(pred_lsds, output_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
                raw_file,
            {
                raw: raw_dataset
            },
            {
                raw: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context,-context)

    prepare_ds(
            out_file,
            lsds_out_ds,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=out_shapes[0])

    prepare_ds(
            out_file,
            affs_out_ds,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True,
            num_channels=out_shapes[1])
    
    model = MtlsdModel(
            config['in_channels'],
            config['output_shapes'],
            config['fmap_inc_factor'],
            config['downsample_factors'],
            config['kernel_size_down'],
            config['kernel_size_up'])
    
    model.eval()

    predict = gp.torch.Predict(
            model,
            checkpoint=os.path.join(setup_dir,f'model_checkpoint_{iteration}'),
            inputs = {
                'input': raw
            },
            outputs = {
                0: pred_lsds,
                1: pred_affs,
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_lsds: lsds_out_ds,
                pred_affs: affs_out_ds
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(raw) +
            gp.Pad(raw, None) +
            gp.IntensityScaleShift(raw, 2,-1) +
            gp.Unsqueeze([raw]) +
            gp.Unsqueeze([raw]) +
            predict +
            gp.Squeeze([pred_lsds,pred_affs]) +
            gp.IntensityScaleShift(pred_lsds, 255, 0) +
            gp.IntensityScaleShift(pred_affs, 255, 0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[raw] = total_input_roi
    predict_request[pred_lsds] = total_output_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

    return total_output_roi


if __name__ == "__main__":

    iterations = [2000,4000,6000,8000,10000,15000,20000]
    raw_file = os.path.join(setup_dir,"../../../../data/test.zarr")
    raw_ds = "raw"
   
    out_file = os.path.join(setup_dir,os.path.basename(raw_file))
    
    for iteration in iterations:

            predict(
                iteration,
                raw_file,
                raw_ds,
                out_file)
