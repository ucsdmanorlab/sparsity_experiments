import os
import neuroglancer
import numpy as np
import networkx as nx
import operator
import sys
import itertools

from funlib.show.neuroglancer import add_layer
from funlib.persistence import open_ds

neuroglancer.set_server_bind_address('localhost',bind_port=3334)


f = sys.argv[1]
skels = sys.argv[2:]

raw = open_ds(f,'raw')
labels = open_ds(f, 'labels')
vs = raw.voxel_size

dims = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=vs)

def to_ng_coords(coords):
    return np.array([x/y for x,y in zip(coords,vs)]).astype(np.float32) + 0.5

def uint64_to_hex_color(value):
    # Ensure the value fits into the range of 0 to 4095
    color_value = value % 4096
    # Convert to base 16 (hexadecimal)
    hex_str = f"{color_value:X}"
    # Pad with 0s to ensure it's at least 3 characters long
    hex_str = hex_str.rjust(3, '0')
    # Convert to the shorthand hex color format by repeating each character
    short_hex_color = '#' + ''.join([c*2 for c in hex_str])
    return short_hex_color


def add_skeletons_from_file(f):
    ngid = itertools.count(start=1)
    
    skeletons = nx.read_graphml(f)
    
    nodes = []
    edges = []

    for u, v in skeletons.edges():
        u = skeletons.nodes[u]
        v = skeletons.nodes[v]

        if 'position_x' not in u or 'position_x' not in v:
           continue

        pos_u = [u['position_z'], u['position_y'], u['position_x']]
        pos_v = [v['position_z'], v['position_y'], v['position_x']]

        pos_u = to_ng_coords(pos_u)
        pos_v = to_ng_coords(pos_v)

        if v['id'] != u['id']:
            print("FFFFFFFFFFFFFFFFFFFFFFF")
            continue

        nodes.append(
                neuroglancer.PointAnnotation(
                    point=pos_u,
                    id=next(ngid),
                    )
                )
        edges.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u,
                    point_b=pos_v,
                    id=next(ngid),
                    )
                )

    return nodes, edges

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, raw, 'raw')
    add_layer(s, labels, 'labels')
    for skel in skels:
        nodes, edges = add_skeletons_from_file(skel) 
        skel_name = os.path.basename(skel).split('.')[0]

        s.layers[f'{skel_name}_nodes'] = neuroglancer.LocalAnnotationLayer(
                #annotation_color='#ff00ff',
                annotations=nodes,
                dimensions=dims,
                #linked_segmentation_layer="labels"
        )
#        s.layers[f'{skel_name}_edges'] = neuroglancer.LocalAnnotationLayer(
#                annotation_color='#add8e6',
#                annotations=edges,
#                dimensions=dims,
#                linked_segmentation_layer="labels"
#                )
print(viewer)
