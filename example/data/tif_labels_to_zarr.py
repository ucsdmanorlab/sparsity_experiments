from tifffile import imread
import numpy as np
import zarr
import sys
import os
from scipy.ndimage import find_objects

""" Makes zarr dataset from napari tif labels. """


if __name__ == "__main__":

    labels_tiff = sys.argv[1] # path to tif file
    #out_zarr = sys.argv[2] # path to zarr container

    ds_name = os.path.basename(labels_tiff).split(".")[0] # get name of dataset from tif file

    if "3d" in ds_name:
        ds_name = "paint_3d"
    else:
        ds_name = "paint_2d"

    out_zarr = os.path.dirname(labels_tiff) # tif file must be inside zarr container

    print(f"writing {ds_name} to {out_zarr}")

    f = zarr.open(out_zarr,"a") 

    # get dataset resolution and offset from raw dataset
    res = f["raw"].attrs["resolution"]
    offset = f["raw"].attrs["offset"]

    # read tif file into numpy array
    labels = imread(labels_tiff)

    print(f"{labels_tiff} has shape {labels.shape}")
    # do bounding box
    slices = find_objects(labels > 0)[0]
    labels = labels[slices]

    # get new offset
    print(offset)
    offset = [offset[i]+(slices[i].start * res[i]) for i in range(3)]
    print(offset)

    # get raw array
    raw = f["raw"][slices]

    assert raw.shape == labels.shape

    # write to zarr
    f[f"{ds_name}/labels"] = labels.astype(np.uint64)
    f[f"{ds_name}/labels"].attrs["offset"] = offset
    f[f"{ds_name}/labels"].attrs["resolution"] = res
    
    f[f"{ds_name}/unlabelled"] = (labels > 0).astype(np.uint8)
    f[f"{ds_name}/unlabelled"].attrs["offset"] = offset
    f[f"{ds_name}/unlabelled"].attrs["resolution"] = res
    
    f[f"{ds_name}/raw"] = raw
    f[f"{ds_name}/raw"].attrs["offset"] = offset
    f[f"{ds_name}/raw"].attrs["resolution"] = res
