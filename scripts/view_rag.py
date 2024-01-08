import os
import neuroglancer
import numpy as np
import networkx as nx
import operator
import sys
import itertools
from pathlib import Path

from funlib.show.neuroglancer import add_layer
from funlib.persistence import open_ds
from funlib.persistence.graphs import SQLiteGraphDataBase

neuroglancer.set_server_bind_address('localhost',bind_port=3334)


f = sys.argv[1]
frags_ds = sys.argv[2]
rag_path = sys.argv[3]
merge_function = sys.argv[4]

frags = open_ds(f,frags_ds)
vs = frags.voxel_size

dims = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="nm", scales=vs)

def to_ng_coords(coords):
    return np.array([x/y for x,y in zip(coords,vs)]).astype(np.float32) + 0.5

def gradient_color(value):
    """
    Returns a color in hexadecimal format as a gradient from green (0) to pink (1).

    :param value: A float between 0 and 1
    :return: Hexadecimal color string
    """
    # Ensure that the input value is within the range [0, 1]
    value = max(0, min(1, value))

    # RGB values for green and pink
    green_rgb = (0, 255, 0)
    pink_rgb = (255, 180, 180)

    # Interpolate between green and pink
    interpolated_rgb = [
        int(green_rgb[i] + (pink_rgb[i] - green_rgb[i]) * value) for i in range(3)
    ]

    # Convert RGB to hexadecimal
    return f"#{interpolated_rgb[0]:02x}{interpolated_rgb[1]:02x}{interpolated_rgb[2]:02x}"


def add_rag_from_file(f):
    ngid = itertools.count(start=1)
    
    rag_provider = SQLiteGraphDataBase(
            Path(f),
            mode="r",
            position_attributes=["position_z", "position_y", "position_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool},
            edges_table="edges_"+merge_function)
    rag = rag_provider.read_graph()

    nodes = []
    edges = []

#    for u,data in rag.nodes(data=True):
#        u = rag.nodes[u]
#        if 'position_x' not in u:
#           continue
#
#        pos_u = [u['position_z'], u['position_y'], u['position_x']]
#        pos_u = to_ng_coords(pos_u)
#        
#        nodes.append(
#                neuroglancer.PointAnnotation(
#                    point=pos_u,
#                    id=next(ngid),
#                    )
#                )

    for u, v, data in rag.edges(data=True):
        u = rag.nodes[u]
        v = rag.nodes[v]

        if 'position_x' not in u or 'position_x' not in v:
           continue

        pos_u = [u['position_z'], u['position_y'], u['position_x']]
        pos_v = [v['position_z'], v['position_y'], v['position_x']]

        pos_u = to_ng_coords(pos_u)
        pos_v = to_ng_coords(pos_v)

        edges.append(
                neuroglancer.LineAnnotation(
                    point_a=pos_u,
                    point_b=pos_v,
                    id=next(ngid),
                    props=[
                        data['merge_score'],
                        gradient_color(data['merge_score']),
                    ]
                    #segments=[u,v]
                    )
                )

    return nodes, edges

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, frags, frags_ds)
    _, edges = add_rag_from_file(rag_path)
    print(f"done reading {len(edges)} edges")
   

#    s.layers['rag_nodes'] = neuroglancer.LocalAnnotationLayer(
#            dimensions=dims,
#            annotations=nodes)

    s.layers['rag_edges'] = neuroglancer.LocalAnnotationLayer(
            dimensions=dims,
            annotation_properties=[
                neuroglancer.AnnotationPropertySpec(id="p_float32",type="float32",default=0),
                neuroglancer.AnnotationPropertySpec(id="color",type="rgb",default="yellow"),
            ],
            annotations=edges,
            shader="""
void main() {
setColor(prop_color());
}
""",
            )
print(viewer)
