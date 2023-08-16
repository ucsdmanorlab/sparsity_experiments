import zarr
import numpy as np
from skimage.morphology import remove_small_objects
import sys

fn = sys.argv[1]
ds = sys.argv[2]

f = zarr.open(fn,"a")

arr = f[ds][:]

if arr.dtype == np.uint64:
    if np.max(arr) > np.iinfo(np.int64).max:
        print("problem")
    else:
        arr = arr.astype(np.int64)

filtered = remove_small_objects(arr, 500)

assert filtered.dtype == arr.dtype
print("before", len(np.unique(arr)))
print("after", len(np.unique(filtered)))

f[f"filtered_{ds}"] = filtered.astype(np.uint64)
f[f"filtered_{ds}"].attrs["offset"] = f[ds].attrs["offset"]
f[f"filtered_{ds}"].attrs["resolution"] = f[ds].attrs["resolution"]
