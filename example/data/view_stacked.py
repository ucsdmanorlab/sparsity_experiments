import neuroglancer
import numpy as np
import os
import sys
import zarr
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


neuroglancer.set_server_bind_address('localhost',bind_port=3334)

f = zarr.open(sys.argv[1]) #input_zarr

#datasets = [i for i in os.listdir(sys.argv[1]) if '.' not in i]
datasets = sys.argv[2:]
print(datasets)

try:

    sections = natural_sort(
            [i for i in os.listdir(os.path.join(sys.argv[1],datasets[0])) if '.' not in i]
            )

    resolution = f[datasets[0]][sections[0]].attrs['resolution']
except:
    resolution = f[datasets[0]]['10'].attrs['resolution']
viewer = neuroglancer.Viewer()

dims = neuroglancer.CoordinateSpace(
        names=['z','y','x'],
        units='nm',
        scales=[8,*resolution])

with viewer.txn() as s:

    for ds in datasets:

        sections = natural_sort(
                [i for i in os.listdir(os.path.join(sys.argv[1],ds)) if '.' not in i]
                )

        data = np.stack([f[ds][section][:] for section in sections])
        try:
            offset = f[ds][sections[0]].attrs['offset']
        except:
            offset = f[ds]['10'].attrs['offset']
        offset = [sections[0]]+[i/vs for i,vs in zip(offset,resolution)]

        volume = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=[*offset],
                dimensions=dims)

        if 'label' in ds or 'id' in ds or 'painted' in ds:
            s.layers[ds] = neuroglancer.SegmentationLayer(source=volume)
        else:
            s.layers[ds] = neuroglancer.ImageLayer(source=volume)

    s.layout = 'yz'

print(viewer)

