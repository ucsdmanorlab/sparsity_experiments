import sys
import os
import numpy as np

from funlib.persistence import open_ds, prepare_ds
from funlib.segment.arrays import replace_values


if __name__ == "__main__":

    frags_file = sys.argv[1]
    frags_dataset = sys.argv[2]
    lut_file = sys.argv[3]
    
    out_file = frags_file
    frag_str = frags_dataset.split('/')[1]
    merge_fn = lut_file.split('/')[-2]
    thresh = os.path.basename(lut_file).split('.')[0]
    out_ds = f"post/{frag_str}/seg/{merge_fn}/{thresh}"

    frags = open_ds(frags_file,frags_dataset)
    vs = frags.voxel_size
    roi = frags.roi
    
    out_seg = prepare_ds(
            out_file,
            out_ds,
            roi,
            vs,
            delete=True,
            dtype=np.uint64)
    
    print("writing seg")
    fragment_segment_lut = np.load(lut_file)
    assert fragment_segment_lut.dtype == np.uint64

    fragments = frags.to_ndarray()
    site_mask = np.isin(fragment_segment_lut[0], fragments)
    seg = replace_values(
        fragments,
        fragment_segment_lut[0][site_mask],
        fragment_segment_lut[1][site_mask])

    out_seg[roi] = seg.astype(np.uint64)
