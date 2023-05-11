import os
import json
import math
import logging
import numpy as np
import gunpowder as gp
import torch
from scipy.ndimage import gaussian_filter

import random
import skimage.draw

from model import AffModel, WeightedMSELoss

logging.basicConfig(level=logging.INFO)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(setup_dir,"config.json")

with open(config_path,"r") as f:
    config = json.load(f)

neighborhood = [[-1,0],[0,-1]]

data_dir = os.path.join(setup_dir,"../../../../data/2d_train.zarr")
sparsity = config["sparsity"] # "" if dense, else "sparsity_crop/rep". example: "obj_002a/rep_3"
if sparsity != "": sparsity += "/"

available_sections = [x for x in os.listdir(os.path.join(data_dir,sparsity+"/labels")) if '.' not in x]
print(f"Available sections to train on: {available_sections}")

def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

    method_padding = gp.Coordinate((sigma * 2,) * 2)

    diag = np.sqrt(output_size[0] ** 2 + output_size[1] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [diag, diag]])
            + method_padding
        ),
        (0,) * 2,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class SmoothArray(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def process(self, batch, request):

        array = batch[self.array].data

        assert len(array.shape) == 2

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(0.0, 1.5)
        
        array = np.array(
                gaussian_filter(array, sigma=sigma)
        ).astype(array.dtype)

        batch[self.array].data = array


class MaskLabels(gp.BatchFilter):

    def __init__(
        self,
        labels,
        output_shape,
        sigma,
        voxel_size,
        scale,
        num_points):

        self.labels = labels
        self.shape = tuple(output_shape)
        self.sigma = sigma
        self.voxel_size = voxel_size
        self.scale = scale
        self.num_points = num_points

    def process(self, batch, request):

        labels = batch[self.labels].data
        shape = labels.shape

        offset = gp.Coordinate(shape) - gp.Coordinate(self.shape)
        offset = tuple([int(x) for x in tuple(offset/2)])

        radius =  gp.Coordinate((self.scale * self.sigma,)*2) / self.voxel_size
        radius = tuple([int(x) for x in radius])

        padded_labels = np.pad(
            labels,
            pad_width=[(radius[0],),(radius[1],)],
            mode='constant',
            constant_values=0)

        mask = np.zeros_like(padded_labels).astype(np.uint8)

        for _ in range(self.num_points):

            x = random.randint(0,self.shape[1] - 1) + radius[1] + offset[1]
            y = random.randint(0,self.shape[0] - 1) + radius[0] + offset[0]

            disk = skimage.draw.disk((x,y), radius=radius[-1], shape=mask.shape)

            mask[disk] = 1

        padded_labels[mask==0] = 0

        padded_labels = padded_labels[
                radius[0]:padded_labels.shape[0]-radius[0],
                radius[1]:padded_labels.shape[1]-radius[1]]

        batch[self.labels].data = padded_labels


class Unlabel(gp.BatchFilter):

  def __init__(self, labels, unlabelled):
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
        voxel_size_3d,
        sigma,
        batch_size,
        **kwargs):

    model = AffModel(
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
        output_shape = model.forward(torch.empty(size=[batch_size,1]+input_shape))[0].shape[1:]
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

    gt_affs = gp.ArrayKey('GT_AFFS')
    gt_affs_mask = gp.ArrayKey('GT_AFFS_MASK')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    affs_weights = gp.ArrayKey('AFFS_WEIGHTS')

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    labels_padding = calc_max_padding(output_size, voxel_size, sigma)

    source = tuple(
        gp.ZarrSource(
            data_dir,
            {
                raw: sparsity + f'raw/{i}',
                labels: sparsity + f'labels/{i}',
                unlabelled: sparsity + f'unlabelled/{i}',
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            }) +
        gp.Normalize(raw) +
        gp.Pad(raw, None) +
        gp.Pad(labels, labels_padding) +
        gp.Pad(unlabelled, labels_padding) +
        gp.RandomLocation(mask=unlabelled,min_masked=0.075)
        for i in available_sections
    )

    pipeline = source
    pipeline += gp.RandomProvider()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(voxel_size_3d[0],) * 2,
        jitter_sigma=(2,2),
        scale_interval=(0.75,1.25),
        rotation_interval=[0,math.pi/2.0],
        subsample=4)

    pipeline += gp.SimpleAugment()

    pipeline += gp.NoiseAugment(raw)

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1)

    pipeline += SmoothArray(raw)

    if 'disk' in setup_dir:
        num_points = int(setup_dir.split('/')[-2].split('-')[0].split('_')[-1])
        print(f"num points in disk sparsity is {num_points}")

        pipeline += MaskLabels(
                labels,
                output_shape,
                sigma,
                voxel_size,
                scale=5,
                num_points=num_points)

        pipeline += Unlabel(labels,unlabelled)
    
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
            0: pred_affs,
        },
        loss_inputs={
            0: pred_affs,
            1: gt_affs,
            2: affs_weights,
        },
        save_every=1000,
        log_dir=os.path.join(setup_dir,'log'),
        checkpoint_basename=os.path.join(setup_dir,'model'))

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
            dataset_names={
                raw: 'raw',
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
