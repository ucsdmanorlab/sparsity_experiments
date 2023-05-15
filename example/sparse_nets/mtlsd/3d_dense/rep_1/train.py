import os
import json
import math
import logging
import numpy as np
import gunpowder as gp
import torch
from lsd.train.gp import AddLocalShapeDescriptor
from scipy.ndimage import gaussian_filter

import random
import skimage.draw

from model import MtlsdModel, WeightedMSELoss

logging.basicConfig(level=logging.INFO)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(setup_dir,"config.json")

with open(config_path,"r") as f:
    config = json.load(f)

neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]

data_dir = os.path.join(setup_dir,"../../../../data/train.zarr")
sparsity = config["sparsity"] # "" if dense, else "sparsity_crop/rep". example: "obj_002a/rep_3"


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class Unlabel(gp.BatchFilter):

  def __init__(sel,,labels, unlabelled):
    self.labels = labels
    self.unlabelled = unlabelled

  def process(self, batch, request):

    labels = batch[self.labels].data

    unlabelled = (labels > 0).astype(np.uint8)

    spec = batch[self.labels].spec.copy()
    spec.roi = request[self.unlabelled].roi.copy()
    spec.dtype = np.uint8

    batch = gp.Batch()

    batch[self.unlabelled] = gp.Array(unlabelled, spec)

    return batch


class SmoothArray(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        array = batch[self.array].data

        assert len(array.shape) == 3

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(0.0, 1.5)

        for z in range(array.shape[0]):
            array_sec = array[z]

            array[z] = np.array(
                    gaussian_filter(array_sec, sigma=sigma)
            ).astype(array_sec.dtype)

        batch[self.array].data = array


def train(
        max_iteration,
        in_channels,
        output_shapes,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        input_shape,
        voxel_size,
        sigma,
        batch_size,
        **kwargs):

    model = MtlsdModel(
            in_channels,
            output_shapes,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    loss = WeightedMSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.5e-4)

    if 'output_shape' not in kwargs:
        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape))[0].shape[2:]
        with open(os.path.join(setup_dir,"config.json"),"r") as f:
            config = json.load(f)
            
        config['output_shape'] = list(output_shape)
            
        with open(os.path.join(setup_dir,"config.json"),"w") as f:
            json.dump(config,f)

    else: output_shape = kwargs.get("output_shape")

    output_shape = gp.Coordinate(tuple(output_shape))
    input_shape = gp.Coordinate(tuple(input_shape))

    voxel_size = gp.Coordinate(voxel_size)
    output_size = output_shape * voxel_size
    input_size = input_shape * voxel_size

    raw = gp.ArrayKey('RAW')
    labels = gp.ArrayKey('LABELS')
    unlabelled = gp.ArrayKey('UNLABELLED')

    gt_lsds = gp.ArrayKey('GT_LSDS')
    pred_lsds = gp.ArrayKey('PRED_LSDS')
    lsds_weights = gp.ArrayKey('LSDS_WEIGHTS')

    gt_affs = gp.ArrayKey('GT_AFFS')
    gt_affs_mask = gp.ArrayKey('GT_AFFS_MASK')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(pred_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    labels_padding = calc_max_padding(output_size, voxel_size, sigma)

    source = gp.ZarrSource(
            data_dir,
            {
                raw: os.path.join(sparsity,"raw"),
                labels: os.path.join(sparsity,"labels"),
                unlabelled: os.path.join(sparsity,"unlabelled"),
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            })
    source += gp.Normalize(raw)
    source += gp.Pad(raw, None)
    source += gp.Pad(labels, labels_padding)
    source += gp.Pad(unlabelled, labels_padding)
    source += gp.RandomLocation(mask=unlabelled,min_masked=0.075)

    pipeline = source

    pipeline += gp.ElasticAugment(
        control_point_spacing=(voxel_size[1],voxel_size[0],voxel_size[0]),
        jitter_sigma=(0,2,2),
        scale_interval=(0.75,1.25),
        rotation_interval=[0,math.pi/2.0],
        subsample=4)

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.NoiseAugment(raw)

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)

    pipeline += SmoothArray(raw)

    pipeline += AddLocalShapeDescriptor(
            labels,
            gt_lsds,
            unlabelled=unlabelled,
            lsds_mask=lsds_weights,
            sigma=sigma,
            downsample=1)
    
    pipeline += gp.AddAffinities(
                affinity_neighborhood=neighborhood,
                labels=labels,
                affinities=gt_affs,
                unlabelled=unlabelled,
                affinities_mask=gt_affs_mask)

    pipeline += gp.BalanceLabels(
                gt_affs,
                affs_weights,
                gt_affs_mask)

    pipeline += gp.IntensityScaleShift(raw, 2,-1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=10, cache_size=40)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            'input': raw
        },
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        save_every=1000,
        log_dir=os.path.join(setup_dir,'log'),
        checkpoint_basename=os.path.join(setup_dir,'model'))

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)
    pipeline += gp.Squeeze([raw,gt_lsds,pred_lsds,lsds_weights,gt_affs,pred_affs,affs_weights])

    pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
                gt_lsds: 'gt_lsds',
                pred_lsds: 'pred_lsds',
                lsds_weights: 'lsds_weights',
                gt_affs: 'gt_affs',
                pred_affs: 'pred_affs',
                affs_weights: 'affs_weights',
            },
            dataset_dtypes={
                gt_affs: np.float32
            },
            output_filename='batch_{iteration}.zarr',
            output_dir=os.path.join(setup_dir,'snapshots'),
            every=1000
    )

    with gp.build(pipeline):
        for i in range(max_iteration):
            batch = pipeline.request_batch(request)


if __name__ == "__main__":

    train(**config)
