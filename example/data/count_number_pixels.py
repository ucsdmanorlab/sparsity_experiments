import os
import sys
import numpy as np
import zarr
import json

""" Reads specified datasets in zarr container, writes out number of pixels in each dataset to json. """

def count_pixels(f,ds):

    arr = zarr.open(f,"r")[ds]

    if "unlabelled" in ds:
        num = int(np.sum(arr))
    else:
        num = int(np.sum(arr[:] > 0))

    return num, int(arr.size)


if __name__ == "__main__":

    f = sys.argv[1]
    dses = sys.argv[2:]

    ret = {}

    for ds in dses:
        nums = count_pixels(f,ds)
        ret[ds] = {"count": nums[0], "size": nums[1]}
        print(ds, nums)

    with open(os.path.join(f,"counts.json"),"w") as f:
        json.dump(ret,f,indent=4)
