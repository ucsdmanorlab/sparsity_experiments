import tqdm
import glob
import sys
import os
from pathlib import Path
import numpy as np
import networkx as nx
import zarr
from funlib.persistence.graphs import SQLiteGraphDataBase
from funlib.persistence import open_ds
from funlib.geometry import Coordinate, Roi

def to_vx_coords(coords,vs):
    return np.array([int((x/y) - 0.5) for x,y in zip(coords,vs)])

def skeletonize(ds):

    f = f"../{ds}/data/train.zarr"
    
    # open labels ds
    labels = open_ds(f,"labels_filtered_relabeled")
    roi = labels.roi
    vs = labels.voxel_size

    # get frags roi
    frags = zarr.open(glob.glob(f"../{ds}/bootstrapped_nets/*/rep_1/train.zarr/small_grid/*/fragments")[0],"r")
    frags_roi = Roi(frags.attrs['offset'],Coordinate(frags.shape)*vs)

    print(ds, roi)
    roi = roi.intersect(frags_roi)
    print(roi)

    out_skel_path = os.path.join(os.path.dirname(f),"waterz_skels_test.db")
    rag_path = os.path.join(f,"post","rag.db")

    # load graph provider
    rag_provider = SQLiteGraphDataBase(
            Path(rag_path),
            mode="r",
            position_attributes=["position_z", "position_y", "position_x"],
            edge_attrs={"merge_score": float, "agglomerated": bool},
            edges_table="edges_mean")

    # read graph
    rag = rag_provider.read_graph(roi=roi)

    # make output nx graph representing skeletons
    skeletons = nx.Graph()

    # load labels array
    labels_arr = labels.to_ndarray(roi)
    labels_ids = np.unique(labels_arr)

    # loop through label masks in label array
    # for label in labels
    for label_id in tqdm.tqdm(labels_ids):
        
        if label_id == 0:
            print('skipping 0')
            continue

        # get label mask
        label_mask = labels_arr == label_id

        sub_skel = nx.Graph()

        # get nodes within label mask
        for u, data in rag.nodes(data=True):

            if 'position_x' not in data:
                continue

            pos_u = Coordinate([data['position_z'], data['position_y'], data['position_x']])

            if roi.contains(pos_u):         
 
                pos_u -= roi.offset
                pos_u = tuple(to_vx_coords(pos_u,vs))

                if label_mask[pos_u] == True:

                    sub_skel.add_node(
                                    u,
                                    id=int(label_id),
                                    position_z=data['position_z'],
                                    position_y=data['position_y'],
                                    position_x=data['position_x'])
       

        # get edges within label mask
        sub_nodes = set(sub_skel.nodes())
        for u,v,data in [(u,v,data) for u,v,data in rag.edges(data=True) if u in sub_nodes and v in sub_nodes]:

            node_u = rag.nodes[u]
            node_v = rag.nodes[v]

            if 'position_x' not in node_u or 'position_x' not in node_v:
                continue

            #print(node_u,node_v)

            pos_u = Coordinate([node_u['position_z'], node_u['position_y'], node_u['position_x']]) - roi.offset
            pos_v = Coordinate([node_v['position_z'], node_v['position_y'], node_v['position_x']]) - roi.offset
            pos_u = to_vx_coords(pos_u,vs)
            pos_v = to_vx_coords(pos_v,vs)

#            # if both nodes are within label mask
#            if label_mask[tuple(pos_u)] == True and label_mask[tuple(pos_v)] == True:

            # calculate distance between u and v
            weight = np.linalg.norm(pos_u - pos_v)

            #sub_skel.add_edge(u,v,u=u,v=v,weight=data['merge_score'])            
            sub_skel.add_edge(u,v,u=u,v=v,weight=weight)            
 
        # MST
        #mst = nx.minimum_spanning_tree(sub_skel,algorithm="prim")
        mst = nx.minimum_spanning_tree(sub_skel)

        # add to output nx with pos attrs, id attr, edge weight attr
        skeletons.update(mst)

    # write output nx to file
    print("writing skel")
    nx.write_graphml(skeletons, out_skel_path) 


if __name__ == "__main__":
    ds = sys.argv[1]

    skeletonize(ds)
