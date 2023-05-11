import os
import gunpowder as gp
import json
import logging
import math
import numpy as np
import random
import torch
import zarr
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label
)
from skimage.morphology import disk
from skimage.measure import label
from model import AffsUNet, WeightedMSELoss
from gunpowder.nodes.add_affinities import seg_to_affgraph

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

def init_weights(m):
    if isinstance(m, (torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')

class ZerosSource(gp.BatchProvider):
    def __init__(self, datasets, shape=None, dtype=np.uint64, array_specs=None):
        self.datasets = datasets

        if array_specs is None:
            self.array_specs = {}
        else:
            self.array_specs = array_specs

        self.shape = shape if shape is not None else gp.Coordinate((200, 200, 200))
        self.dtype = dtype

        # number of spatial dimensions
        self.ndims = None

    def setup(self):
        for array_key, ds_name in self.datasets.items():
            if array_key in self.array_specs:
                spec = self.array_specs[array_key].copy()
            else:
                spec = gp.ArraySpec()

            if spec.voxel_size is None:
                voxel_size = gp.Coordinate((1,) * len(self.shape))
                spec.voxel_size = voxel_size

            self.ndims = len(spec.voxel_size)

            if spec.roi is None:
                offset = gp.Coordinate((0,) * self.ndims)
                spec.roi = gp.Roi(offset, self.shape * spec.voxel_size)

            if spec.dtype is not None:
                assert spec.dtype == self.dtype
            else:
                spec.dtype = self.dtype

            if spec.interpolatable is None:
                spec.interpolatable = spec.dtype in [
                    np.float,
                    np.float32,
                    np.float64,
                    np.float10,
                    np.uint8,  # assuming this is not used for labels
                ]

            self.provides(array_key, spec)


    def provide(self, request):
        batch = gp.Batch()

        for array_key, request_spec in request.array_specs.items():
            voxel_size = self.spec[array_key].voxel_size

            # scale request roi to voxel units
            dataset_roi = request_spec.roi / voxel_size

            # shift request roi into dataset
            dataset_roi = (
                dataset_roi - self.spec[array_key].roi.get_offset() / voxel_size
            )

            # create array spec
            array_spec = self.spec[array_key].copy()
            array_spec.roi = request_spec.roi
            
            # add array to batch
            batch.arrays[array_key] = gp.Array(
                np.zeros(self.shape, self.dtype)[dataset_roi.to_slices()], array_spec
            )

        return batch

class CreateLabels(gp.BatchFilter):
    def __init__(
        self,
        labels,
        anisotropy
    ):

        self.labels = labels
        self.anisotropy = anisotropy + 1

    def process(self, batch, request):

        labels = batch[self.labels].data
        labels = np.concatenate([labels,]*self.anisotropy)
        shape = labels.shape

        # different numbers simulate more or less objects
        num_points = random.randint(25,50*self.anisotropy)

        for n in range(num_points):
            z = random.randint(1, labels.shape[0] - 1)
            y = random.randint(1, labels.shape[1] - 1)
            x = random.randint(1, labels.shape[2] - 1)

            labels[z, y, x] = 1

        structs = [generate_binary_structure(2, 2), disk(random.randint(1,5))]

        #different numbers will simulate larger or smaller objects

        for z in range(labels.shape[0]):

            struct = random.choice(structs)
            dilations = random.randint(1, 10)

            dilated = binary_dilation(
                labels[z], structure=struct, iterations=dilations
            )

            labels[z] = dilated.astype(labels.dtype)

        #relabel
        labels = label(labels)

        #expand labels
        distance = labels.shape[0]

        distances, indices = distance_transform_edt(
            labels == 0, return_indices=True
        )

        expanded_labels = np.zeros_like(labels)

        dilate_mask = distances <= distance

        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]

        nearest_labels = labels[tuple(masked_indices)]

        expanded_labels[dilate_mask] = nearest_labels

        labels = expanded_labels

        #change background
        labels[labels == 0] = np.max(labels) + 1

        #relabel
        labels = label(labels)

        batch[self.labels].data = labels[::self.anisotropy].astype(np.uint64)


class SmoothAffs(gp.BatchFilter):
    def __init__(self, affs):
        self.affs = affs

    def process(self, batch, request):

        affs = batch[self.affs].data

        # different numbers will simulate noisier or cleaner affs
        sigma = random.uniform(0.5, 1.5)

        for z in range(affs.shape[1]):
            affs_sec = affs[:, z]

            affs[:, z] = np.array(
                [
                    gaussian_filter(affs_sec[i], sigma=sigma)
                    for i in range(affs_sec.shape[0])
                ]
            ).astype(affs_sec.dtype)

        batch[self.affs].data = affs


class CustomAffs(gp.BatchFilter):
    def __init__(
            self,
            affinity_neighborhood,
            labels,
            affinities,
            dtype=np.float32):

        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.affinities = affinities
        self.dtype = dtype

    def setup(self):

        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by "
            "AddAffinities"%self.labels)

        voxel_size = self.spec[self.labels].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = gp.Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        self.padding_pos = gp.Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
        )*voxel_size

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = self.dtype

        self.provides(self.affinities, spec)
        self.enable_autoskip()

    def prepare(self, request):

        deps = gp.BatchRequest()

        # grow labels ROI to accomodate padding
        labels_roi = request[self.affinities].roi.grow(
            -self.padding_neg,
            self.padding_pos)
        deps[self.labels] = request[self.affinities].copy()
        deps[self.labels].dtype = None
        deps[self.labels].roi = labels_roi

        return deps

    def process(self, batch, request):
        outputs = gp.Batch()

        affinities_roi = request[self.affinities].roi

        affinities = seg_to_affgraph(
            batch.arrays[self.labels].data.astype(np.int32),
            self.affinity_neighborhood
        ).astype(self.dtype)

        # crop affinities to requested ROI
        offset = affinities_roi.get_offset()
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        affinities = affinities[(slice(None),)+crop]

        # remove z-channel of affinities
        affinities = affinities[1:3]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        outputs.arrays[self.affinities] = gp.Array(affinities, spec)

        # Should probably have a better way of handling arbitrary batch attributes
        batch.affinity_neighborhood = self.affinity_neighborhood[1:]

        return outputs


def train(
        iterations,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        input_shape,
        voxel_size,
        batch_size,
        neighborhood,
        **kwargs):

    model = AffsUNet(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up)

    #model.apply(init_weights)

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

    zeros = gp.ArrayKey("ZEROS")
    affs_gt = gp.ArrayKey("AFFS_2D")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    if 'output_shape' not in kwargs:
        output_shape = model.forward(torch.empty(size=[batch_size,in_channels]+input_shape))[0].shape[1:]
        with open(os.path.join(setup_dir,"config.json"),"r") as f:
            config = json.load(f)

        config['output_shape'] = list(output_shape)

        with open(os.path.join(setup_dir,"config.json"),"w") as f:
            json.dump(config,f)

    else: output_shape = kwargs.get("output_shape")

    output_shape = gp.Coordinate(tuple(output_shape))
    input_shape = gp.Coordinate(tuple(input_shape))

    voxel_size = gp.Coordinate(voxel_size)
    
    anisotropy = int((voxel_size[0] / voxel_size[1]) - 1) # 0 is isotropic

    output_size = output_shape * voxel_size
    input_size = input_shape * voxel_size

    request = gp.BatchRequest()

    request.add(zeros, input_size)
    request.add(affs_gt, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    source = ZerosSource(
        {
            zeros: "zeros",  # just a zeros dataset, since we need a source
        },
        shape=input_shape,
        array_specs={
            zeros: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
        },
    )

    source += gp.Pad(zeros, None)

    pipeline = source

    pipeline += CreateLabels(zeros,anisotropy)

    if input_shape[0] != input_shape[1]:
        pipeline += gp.SimpleAugment(transpose_only=[1, 2])
    else:
        pipeline += gp.SimpleAugment()

    pipeline += gp.ElasticAugment(
        control_point_spacing=[voxel_size[1], voxel_size[0], voxel_size[0]],
        jitter_sigma=[2*int(not bool(anisotropy)), 2, 2],
        scale_interval=(0.75,1.25),
        rotation_interval=[0,math.pi/2.0],
        subsample=4,
    )

    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(zeros, steps=1, only_xy=bool(anisotropy))

    # do this on non eroded labels - that is what predicted affs will look like
    pipeline += CustomAffs(
        affinity_neighborhood=neighborhood,
        labels=zeros,
        affinities=affs_gt,
        dtype=np.float32,
    )

    # add random noise
    pipeline += gp.NoiseAugment(affs_gt)
    
    pipeline += gp.IntensityAugment(affs_gt, 0.9, 1.1, -0.1, 0.1)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothAffs(affs_gt)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=zeros,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": affs_gt},
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=1000,
        checkpoint_basename=os.path.join(setup_dir,'model'),
        log_dir=os.path.join(setup_dir,'log')
    )

    pipeline += gp.Squeeze([affs_gt,gt_affs,pred_affs])
    
    pipeline += gp.Snapshot(
        dataset_names={
            zeros: "labels",
            affs_gt: "affs_gt",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir,'snapshots'),
        every=1000,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    config_path = os.path.join(setup_dir,"config.json")

    with open(config_path,"r") as f:
        config = json.load(f)

    train(**config)
