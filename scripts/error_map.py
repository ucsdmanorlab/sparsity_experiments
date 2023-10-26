import gunpowder as gp
from funlib.segment.graphs import find_connected_components
from funlib.segment.arrays import replace_values
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

from lsd.train.gp import AddLocalShapeDescriptor

import json
import logging
import sys
import numpy as np


class ComputeDistance(gp.BatchFilter):

    def __init__(self, a, b, diff, mask):

        self.a = a
        self.b = b
        self.diff = diff
        self.mask = mask

    def setup(self):

        self.provides(
            self.diff,
            self.spec[self.a].copy())

    def prepare(self, request):

        request[self.a] = request[self.diff].copy()
        request[self.b] = request[self.diff].copy()
        request[self.mask] = request[self.diff].copy()

    def process(self, batch, request):

        a_data = batch[self.a].data
        b_data = batch[self.b].data
        mask_data = batch[self.mask].data

        diff_data = np.sum((a_data - b_data)**2, axis=0)
        diff_data *= mask_data

        # normalize
        epsilon = 1e-10  # a small constant to avoid division by zero
        max_value = np.max(diff_data)

        if max_value > epsilon:
            diff_data /= max_value
        else:
            diff_data[:] = 0

        batch[self.diff] = gp.Array(
            diff_data,
            batch[self.a].spec.copy())


def compute_error_map(
        seg_file,
        seg_dataset,
        labels_file,
        labels_dataset,
        mask_dataset,
        diffs_file,
        diffs_dataset,
        **kwargs):
 
    segmentation = gp.ArrayKey('SEGMENTATION')
    labels = gp.ArrayKey('LABELS')
    mask = gp.ArrayKey('MASK')
    seg_pred = gp.ArrayKey('SEG_PRED')
    gt_pred = gp.ArrayKey('GT_PRED')
    pred_diff = gp.ArrayKey('PRED_DIFF')

    # get rois
    labels_roi = open_ds(labels_file,labels_dataset).roi
    seg_roi = open_ds(seg_file,seg_dataset).roi
    roi = labels_roi.intersect(seg_roi)
    total_input_roi = total_output_roi = roi 

    increase = Coordinate((0,80,80)) * 0
    input_shape = Coordinate((6,128,128)) + increase
    output_shape = Coordinate((4,108,108)) + increase

    print(f"seg roi: {seg_roi}, labels_roi: {labels_roi}, intersection: {roi}")
    voxel_size = Coordinate((50,8,8))
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    chunk_request = gp.BatchRequest()
    chunk_request.add(segmentation, input_size)
    chunk_request.add(labels, input_size)
    chunk_request.add(pred_diff, output_size)

    prepare_ds(
        diffs_file,
        diffs_dataset,
        total_output_roi,
        voxel_size,
        np.float32,
        write_size=output_size,
        delete=True) 

    sources = (
        gp.ZarrSource(
                seg_file,
                datasets = {
                    segmentation: seg_dataset
                },
                array_specs = {
                    segmentation: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False, roi=seg_roi)
                }
            ),
        gp.ZarrSource(
                labels_file,
                datasets = {
                    labels: labels_dataset,
                    mask: mask_dataset
                },
                array_specs = {
                    labels: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False, roi=labels_roi),
                    mask: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False, roi=labels_roi),
                }
            )
    )

    pipeline = sources + gp.MergeProvider()

    pipeline += gp.Pad(segmentation, size=None)
    pipeline += gp.Pad(labels, size=None)
    pipeline += gp.Pad(mask, size=None)

#    neighborhood = [[-1,0,0],[0,-1,0],[0,0,-1]]
#
#    pipeline += gp.AddAffinities(
#        affinity_neighborhood=neighborhood,
#        labels=segmentation,
#        affinities=seg_affs)
#
#    pipeline += gp.AddAffinities(
#        affinity_neighborhood=neighborhood,
#        labels=labels,
#        affinities=gt_affs)
#

#    pipeline += gp.GrowBoundary(segmentation,only_xy=True)

    pipeline += AddLocalShapeDescriptor(
        segmentation,
        seg_pred,
        sigma=80,
        downsample=1)
    
    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_pred,
        sigma=80,
        downsample=1)

    pipeline += ComputeDistance(
        seg_pred,
        gt_pred,
        pred_diff,
        mask)

    pipeline += gp.ZarrWrite(
            dataset_names={
                pred_diff: diffs_dataset
            },
            store=diffs_file
        )
    pipeline += gp.Scan(chunk_request)

    full_request = gp.BatchRequest()
    full_request[segmentation] = total_input_roi
    full_request[labels] = total_input_roi
    full_request[pred_diff] = total_output_roi

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(full_request)
    print("Prediction finished")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    f = '/scratch/04101/vvenu/sparsity_experiments/voljo/data/train.zarr'
    #datasets = ['affs_mtlsd/obj_001/rep_2/seg', 'affs/obj_010/rep_1/seg', 'affs/obj_100/rep_3/seg', 'affs_mtlsd/10min_paint_2d/rep_3/seg', 'affs/paint_2d/rep_1/seg', 'affs_mtlsd/10min_paint_3d/rep_3/seg', 'affs/paint_3d/rep_2/seg', 'affs/2d_dense/rep_1/seg', 'lsds_mtlsd/3d_dense/rep_1/seg']
    dses = ['affs/obj_001/rep_2/seg', 'affs/obj_010/rep_2/seg', 'affs_mtlsd/obj_100/rep_1/seg', 'affs_mtlsd/10min_paint_2d/rep_3/seg', 'affs/paint_2d/rep_1/seg', 'affs_mtlsd/10min_paint_3d/rep_3/seg', 'affs_mtlsd/paint_3d/rep_3/seg', 'affs/2d_dense/rep_3/seg', 'lsds_mtlsd/3d_dense/rep_1/seg', 'gt_dense/gt_dense/rep_3/seg']
    datasets = []
    for x in dses:
        if 'gt_dense' in x:
            x = f"BS-gt_dense/{x.split('/')[2]}/seg"
        else:
            x = f"BS-{x.split('/')[0]}-{x.split('/')[1]}/{x.split('/')[2]}/seg"
        datasets.append(x)

    #datasets = [datasets[5]]
    print(datasets)

    for ds in datasets:
        print(ds)
        compute_error_map(
            seg_file=f,
            seg_dataset=ds,
            labels_file="/scratch/04101/vvenu/sparsity_experiments/voljo/data/FRESH/apical.zarr",
            labels_dataset="labels/s2",
            mask_dataset="labels_mask/s2",
            diffs_file="/scratch/04101/vvenu/sparsity_experiments/voljo/newBS_errors.zarr",
            diffs_dataset=ds.split('/')[0].split('-')[-1], 
        )
