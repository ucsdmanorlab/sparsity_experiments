import json
import gunpowder as gp
import math
import numpy as np
import os
import sys
import torch
import logging
import zarr
import daisy
from funlib.persistence import prepare_ds

from model import AffsUNet

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)

# voxels
increase = gp.Coordinate([16,80,80])
input_shape = gp.Coordinate(tuple(config['input_shape'])) + increase
output_shape = gp.Coordinate(tuple(config['output_shape'])) + increase

# nm
voxel_size = gp.Coordinate(tuple(config['voxel_size']))
input_size = input_shape * voxel_size
output_size = output_shape * voxel_size
context = (input_size - output_size) / 2


def predict(
        iteration,
        input_affs_file,
        input_affs_dataset,
        out_file):

    network, sparsity, rep = input_affs_file.split('/')[-4:-1]
    out_ds = f"{network}/{sparsity}/{rep}/3d_affs_{iteration}_from_{input_affs_dataset}"

    model = AffsUNet(
            config['in_channels'],
            config['num_fmaps'],
            config['fmap_inc_factor'],
            config['downsample_factors'],
            config['kernel_size_down'],
            config['kernel_size_up'])
    
    model.eval()

    input_affs = gp.ArrayKey('INPUT_AFFS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    scan_request = gp.BatchRequest()

    scan_request.add(input_affs, input_size)
    scan_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
                input_affs_file,
            {
                input_affs: input_affs_dataset
            },
            {
                input_affs: gp.ArraySpec(interpolatable=True)
            })

    with gp.build(source):
        total_input_roi = source.spec[input_affs].roi
        total_output_roi = source.spec[input_affs].roi.grow(-context,-context)

    prepare_ds(
            out_file,
            out_ds,
            daisy.Roi(
                total_output_roi.get_offset(),
                total_output_roi.get_shape()
            ),
            voxel_size,
            np.uint8,
            write_size=output_size,
            compressor={"id":"blosc","clevel":3},
            delete=True,
            num_channels=len(config['neighborhood']))

    predict = gp.torch.Predict(
            model,
            checkpoint=os.path.join(setup_dir,f'model_checkpoint_{iteration}'),
            inputs = {
                'input': input_affs
            },
            outputs = {
                0: pred_affs,
            })

    scan = gp.Scan(scan_request)

    write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_ds,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(input_affs) +
            gp.Pad(input_affs, None) +
            gp.Unsqueeze([input_affs]) +
            predict +
            gp.Squeeze([pred_affs]) +
            gp.IntensityScaleShift(pred_affs,255,0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    predict_request[input_affs] = total_input_roi
    predict_request[pred_affs] = total_output_roi

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)

    return total_output_roi

if __name__ == "__main__":

    iterations = [2000,3000,4000,5000,6000,8000,10000,15000,20000]
    input_affs_file = sys.argv[1]
    input_affs_dataset = sys.argv[2]

    out_file = os.path.join(setup_dir,os.path.basename(input_affs_file))

    for iteration in iterations:

        total_output_roi = predict(
            iteration,
            input_affs_file,
            input_affs_dataset,
            out_file)
