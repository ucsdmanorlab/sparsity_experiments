import zarr
import sys
import numpy as np
from scipy.ndimage import find_objects

def partition(nums, num_partitions):
  
    # Sort descending
    nums = sorted(nums, key = lambda x: x[1], reverse=True)

    # Initialize the partitions as empty lists
    partitions = [[] for _ in range(num_partitions)]

    # Initialize the sums of the partitions
    sums = [0] * num_partitions

    # Iterate through the numbers and add them to the partitions
    for num in nums:
        # Skip 0
        if num[0] == 0: continue

        # Find the partition with the smallest sum
        min_sum = min(sums)

        # Add the number to the partition with the smallest sum
        min_sum_idx = sums.index(min_sum)
        partitions[min_sum_idx].append(num)
        sums[min_sum_idx] += num[1]

    return partitions


if __name__ == "__main__":

    zarr_dir = sys.argv[1]
    labels_ds = sys.argv[2]
    num_partitions = int(sys.argv[3])

    # Open zarr
    f = zarr.open(zarr_dir,"a")
    labels = f[labels_ds+"/labels"]
    offset = labels.attrs["offset"]
    res = labels.attrs["resolution"]

    labels_arr = labels[:]
    raw_arr = f[labels_ds+"/raw"][:]

    # Get uniques and counts of labels
    uniques, counts = np.unique(labels_arr, return_counts=True)
    pairs = list(zip(uniques,counts))

    partitions = partition(pairs, num_partitions)

    for i, part in enumerate(partitions):

        uniques = [x[0] for x in part]
        new_ds = f"10min_{labels_ds}/rep_{i+1}"

        # Mask in selected labels
        new_arr = labels_arr.copy()
        new_arr[np.isin(new_arr, uniques, invert=True)] = 0

        # New offset
        slices = find_objects(new_arr > 0)[0]
        new_offset = [offset[i]+(slices[i].start * res[i]) for i in range(3)]

        # Write
        f[new_ds+"/labels"] = new_arr[slices]
        f[new_ds+"/labels"].attrs["offset"] = new_offset
        f[new_ds+"/labels"].attrs["resolution"] = res
        
        f[new_ds+"/unlabelled"] = (new_arr[slices] > 0).astype(np.uint8)
        f[new_ds+"/unlabelled"].attrs["offset"] = new_offset
        f[new_ds+"/unlabelled"].attrs["resolution"] = res
        
        f[new_ds+"/raw"] = raw_arr[slices]
        f[new_ds+"/raw"].attrs["offset"] = new_offset
        f[new_ds+"/raw"].attrs["resolution"] = res
