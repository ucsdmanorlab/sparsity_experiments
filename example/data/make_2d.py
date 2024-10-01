import numpy as np
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi
import sys
import os
from multiprocessing import Pool


""" Script to convert a zarr container with 3D datasets to a new container with 2D datasets. """


num_workers = 1 # write sections in parallel


def write_section(args):

    data = args['data']
    index = args['index']

    section_number = int(roi.offset[0]/vs[0] + index)
    
    if np.any(data):

        print(f"at section {section_number}..")

        new_ds = prepare_ds(
                output_zarr,
                f"{dataset}/{section_number}",
                write_roi,
                write_vs,
                dtype,
                compressor=dict(id='blosc'))

        new_ds[write_roi] = data

    else:
        print(f"section {section_number} is empty, skipping")
        pass


if __name__ == "__main__":

    input_zarr = sys.argv[1]

    output_zarr = os.path.join(os.path.dirname(input_zarr),"2d_"+os.path.basename(input_zarr))

    #datasets = [i for i in os.listdir(sys.argv[1]) if '.z' not in i] # all datasets in input_zarr
    
    datasets = sys.argv[2:]
    print(datasets)

    for dataset in datasets:

        ds = open_ds(input_zarr,dataset)

        data = ds.to_ndarray()

        roi = ds.roi
        vs = ds.voxel_size
        dtype = ds.dtype
 
        write_roi = Roi(roi.offset[1:],roi.shape[1:])
        write_vs = vs[1:]

        args = ({
                'index' : index,
                'data' : section} for index,section in enumerate(data))

        with Pool(num_workers,maxtasksperchild=1) as pool:

           pool.map(write_section,args) 
