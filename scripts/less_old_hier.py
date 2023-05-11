from math import floor
import numpy as np
import daisy

from funlib.geometry import Roi, Coordinate

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.transform import rescale, downscale_local_mean

from lsd.post.fragments import watershed_from_affinities
import waterz


""" Script to do hierarchical segmentation of 2D or 3D direct neighbor affinities. """


waterz_merge_function = {
    'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
    'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
    'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
    'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
    'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
    'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
    'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
    'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
    'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
    'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
    'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
}


def process(
        pred,
        downsampling,
        denoising,
        normalize_preds):


    d = len(pred.shape) - 3

    #downsample
    if downsampling is not None:

        if downsampling[0] == 'rescale':

            pred = rescale(
                pred,
                [1,]*d + [1,1/downsampling[1],1/downsampling[1]],
                anti_aliasing=True,
                order=1)
            pred = pred.astype(np.float32)

        elif downsampling[0] == 'local_mean':

            pred = downscale_local_mean(pred,(1,)*d + (1,downsampling[1],downsampling[1]))
            pred = pred.astype(np.float32)

        else:

            pred = pred[:,:,::downsampling[1],::downsampling[1]] if d==1 else pred[:,::downsampling[1],::downsampling[1]]

    #denoise
    if denoising is not None:

        if denoising[0] == "tv":

            pred = denoise_tv_chambolle(
                pred,
                weight=denoising[1],
                channel_axis=0)
            pred = pred.astype(np.float32)

        elif denoising[0] == "bilateral":

            if pred.dtype == np.uint8:
                pred = pred.astype(np.float32)/255.0

            if d == 1:
                for s in range(pred.shape[1]):

                    pred[:,s] = denoise_bilateral(
                        pred[:,s],
                        sigma_color=denoising[1],
                        sigma_spatial=denoising[2],
                        channel_axis=0)
            else:
                pred = denoise_bilateral(
                    pred,
                    sigma_color=denoising[1],
                    sigma_spatial=denoising[2],
                    channel_axis=0)

            pred = pred.astype(np.float32)

        else:

            raise KeyError("unknown denoising mode for preds")

    else:
        pred = pred.astype(np.float32)/255.0 if pred.dtype == np.uint8 else pred

    #normalize channel-wise
    if normalize_preds:
        for c in range(len(pred)):
            
            max_v = np.max(pred[c])
            min_v = np.min(pred[c])

            if max_v != min_v:
                pred[c] = (pred[c] - min_v)/(max_v - min_v)
            else:
                pred[c] = np.ones_like(pred[c])
    
    return pred


def post(
        pred_file,
        pred_dataset,
        roi,
        downsampling,
        denoising,
        normalize_preds,
        background_mask,
        min_seed_distance,
        merge_function,
        write=False):


    #load
    pred = daisy.open_ds(pred_file,pred_dataset)
    
    if roi is not None:
        roi = daisy.Roi(pred.roi.offset+daisy.Coordinate(roi[0]),roi[1])
    else:
        roi = pred.roi

    if write:
        vs = pred.voxel_size

    pred = pred.to_ndarray(roi)
    pred = np.expand_dims(pred,1) if len(pred.shape) == 3 else pred
    pred = pred.astype(np.float32)/255.0 if pred.dtype == np.uint8 else pred

    #first three channels are direct neighbor affs
    if len(pred) > 3:
        pred = pred[:3]
    
    #process
    pred = process(
            pred,
            downsampling,
            denoising,
            normalize_preds)
    
    #make fragments
    frags = watershed_from_affinities(
        pred.copy(),
        background_mask=background_mask,
        fragments_in_xy=True,
        min_seed_distance=min_seed_distance)[0]

    #agglomerate
    max_thresh = 1.0
    step = 1/50
    
    thresholds = [round(x,2) for x in np.arange(0,max_thresh,step)]

    segs = {}

    generator = waterz.agglomerate(
            pred,
            thresholds=thresholds,
            fragments=frags.copy(),
            scoring_function=waterz_merge_function[merge_function])

    for threshold,segmentation in zip(thresholds,generator):

        #upsample seg
        if downsampling is not None:
            seg = rescale(segmentation.copy(), [1,downsampling[1],downsampling[1]], order=0)
        else:
            seg = segmentation.copy()
        
        segs[threshold] = seg

    #upsample
    if downsampling is not None:

        frags = rescale(frags, [1,downsampling[1],downsampling[1]], order=0)
        pred = rescale(pred, [1,1,downsampling[1],downsampling[1]], order=1)

    #write
    if write:
        roi = daisy.Roi(roi.offset,Coordinate(seg.shape)*vs)

        out_seg = daisy.prepare_ds(
                pred_file,
                "segmentation_0.5",
                roi,
                vs,
                np.uint64,
                num_channels=1)

        out_seg[roi] = np.expand_dims(segs[0.5],axis=0)

    else:
        return segs, frags, pred
