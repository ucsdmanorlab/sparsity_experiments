import os
import sys
import numpy as np
import zarr
import json
from scipy.ndimage import find_objects

""" Reads specified datasets in zarr container, writes out number of pixels in each dataset to json. """

def count_pixels(f,ds):

    arr = zarr.open(f,"r")[ds][:]

    bbox = find_objects(arr > 0)[0]
    cropped = arr[bbox]

    num = int(np.sum(cropped > 0))

    print(f,ds, num, int(cropped.size), len(np.unique(cropped))) 
    return num, int(cropped.size), len(np.unique(cropped)) 


if __name__ == "__main__":

    fs = sys.argv[1:]

    dses = [
        'labels',
        '10min_paint_2d/rep_1/labels',
        '10min_paint_2d/rep_2/labels',
        '10min_paint_2d/rep_3/labels',
        'paint_2d/labels',
        '10min_paint_3d/rep_1/labels',
        '10min_paint_3d/rep_2/labels',
        '10min_paint_3d/rep_3/labels',
        'paint_3d/labels',
        'obj_001/rep_1/labels',
        'obj_001/rep_2/labels',
        'obj_001/rep_3/labels',
        'obj_002/rep_1/labels',
        'obj_002/rep_2/labels',
        'obj_002/rep_3/labels',
        'obj_002a/rep_1/labels',
        'obj_002a/rep_2/labels',
        'obj_002a/rep_3/labels',
        'obj_005/rep_1/labels',
        'obj_005/rep_2/labels',
        'obj_005/rep_3/labels',
        'obj_010/rep_1/labels',
        'obj_010/rep_2/labels',
        'obj_010/rep_3/labels',
        'obj_050/rep_1/labels',
        'obj_050/rep_2/labels',
        'obj_050/rep_3/labels',
        'obj_100/rep_1/labels',
        'obj_100/rep_2/labels',
        'obj_100/rep_3/labels'
    ]

    for f in fs:
        ret = {}

        N = count_pixels(f,"labels")[0]

        for ds in dses:
            if "rep" in ds:
                nums = count_pixels(f,ds)
                ret[ds] = {"labels": nums[2] - 1, "count": nums[0], "size": nums[1], "ratio": nums[0]/N}
            else:
                nums = count_pixels(f,ds)
                for r in ["rep_1","rep_2","rep_3"]:
                    ret[ds+f"/{r}"] = {"labels": nums[2] - 1, "count": nums[0], "size": nums[1], "ratio": nums[0]/N}

        with open(os.path.join(f,"counts.json"),"w") as out:
            json.dump(ret,out,indent=4)
