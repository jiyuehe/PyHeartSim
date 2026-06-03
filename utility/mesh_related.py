# Copyright 2026 Jiyue He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# find neighbor vertices for each vertex
def find_neighbor_vertices(n_vertices, face):
    # build all directed edges from faces. for each triangle, there are 3 edges each has 2 directions, a total of 6 pairs of vertices
    edges = np.concatenate([face[:, [0, 1]], face[:, [1, 2]], face[:, [0, 2]], face[:, [1, 0]], face[:, [2, 1]], face[:, [2, 0]]])
    edges = np.unique(edges, axis=0) # sort and remove duplicate
    # edges[0:20] =
    # array([[    0,  9389],
    #     [    0, 37372],
    #     [    0, 57990],
    #     [    0, 58189],
    #     [    0, 58202],
    #     [    0, 87468],
    #     [    1,  9415],
    #     [    1, 37198],
    #     [    1, 37199],
    #     [    1, 37384],
    #     [    1, 57998],
    #     [    1, 58212],
    #     [    2,  9193],
    #     [    2,  9197],
    #     [    2, 37203],
    #     [    2, 57993],
    #     [    2, 58003],
    #     [    2, 82737],
    #     [    3,  9199],
    #     [    3, 37201]])

    splits = np.searchsorted(edges[:, 0], np.arange(n_vertices + 1))
    # splits = [ 0, 6, 12, ...]
    
    neighbor_vertices_ids = [edges[splits[i]:splits[i+1], 1] for i in range(n_vertices)]
    # edges[splits[0]:splits[0+1], 1] = array([ 9389, 37372, 57990, 58189, 58202, 87468])

    return neighbor_vertices_ids

def identify_tip_of_pulmonary_veins(vertex, face, neighbor_vertices_ids):
    n_vertices = vertex.shape[0]

    # compute the distance from each vertex to the center of mass
    center_of_mass = np.mean(vertex, axis=0)
    vertex_to_COM_distance = np.linalg.norm(vertex - center_of_mass, axis=1)

    # for each vertex, find the trail to the "highest" vertex
    vertex_id_trail = []
    for v_id in range(n_vertices):
        if (v_id + 1) % (n_vertices // 10) == 0:
            print(f"{(v_id + 1) / n_vertices * 100:.2f}%")

        # greedy ascent: follow neighbors with increasing distance from center of mass until no neighbor has a larger distance
        trail = [v_id] # start from the current vertex
        current = v_id
        while True:
            nbrs = np.asarray(neighbor_vertices_ids[current], dtype=int)

            max_idx = np.argmax(vertex_to_COM_distance[nbrs])

            if vertex_to_COM_distance[nbrs[max_idx]] > vertex_to_COM_distance[current]:
                current = int(nbrs[max_idx])
                trail.append(current)
            else:
                break

        vertex_id_trail.append(trail)

    debug_plot = 0
    if debug_plot == 1: # show an example of the trail
        # find the longest trail
        longest_trail_idx = np.argmax([len(trail) for trail in vertex_id_trail])
        trail = vertex_id_trail[longest_trail_idx]

        vertex_color = np.ones((n_vertices, 3))
        vertex_color[trail[-1], :] = [1, 0, 0] # final vertex in red
        for i in range(len(trail) - 1):
            vertex_color[trail[i], :] = [0, 0, 1] # trail vertices in blue

        # build face list for Mesh3d
        fi, fj, fk = face[:, 0], face[:, 1], face[:, 2]

        vertex_color_str = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in vertex_color]

        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                i=fi, j=fj, k=fk,
                color='white', opacity=1.0,
                lighting=dict(ambient=1.0, diffuse=0, specular=0, roughness=1, fresnel=0),
                flatshading=True,
            ),

            # triangle edges: interleave [A, B, C, A, None] per triangle
            go.Scatter3d(
                x=np.stack([vertex[fi, 0], vertex[fj, 0], vertex[fk, 0], vertex[fi, 0], np.full(len(fi), None)], axis=1).ravel(),
                y=np.stack([vertex[fi, 1], vertex[fj, 1], vertex[fk, 1], vertex[fi, 1], np.full(len(fi), None)], axis=1).ravel(),
                z=np.stack([vertex[fi, 2], vertex[fj, 2], vertex[fk, 2], vertex[fi, 2], np.full(len(fi), None)], axis=1).ravel(),
                mode='lines',
                line=dict(color='gray', width=1),
            ),

            go.Scatter3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                mode='markers',
                marker=dict(size=2, color=vertex_color_str),
            ),
        ])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show() # opens in browser

    # find the "highest" vertices
    highest_vertex_id_of_each_trail = np.zeros(len(vertex_id_trail), dtype=int)
    for i, trail in enumerate(vertex_id_trail):
        highest_vertex_id_of_each_trail[i] = trail[-1]

    highest_vertex_id = np.unique(highest_vertex_id_of_each_trail)
    highest_vertex = vertex[highest_vertex_id, :]

    # give labels to the each vertex based on the highest_vertex it leads to. vertices leading to the same highest_vertex will have the same label
    vertex_labels = np.zeros(n_vertices, dtype=int)
    for i, trail in enumerate(vertex_id_trail):
        vertex_labels[trail] = trail[-1]

    # cluster the highest_vertex based on their spatial proximity
    distanceThreshold = 25  # unit: mm
    pairwise_distances = distance.pdist(highest_vertex) # condensed distance matrix
    Z = linkage(pairwise_distances, method='single')
    cluster_labels = fcluster(Z, t=distanceThreshold, criterion='distance')

    # for each cluster, find the vertex with the largest distance to center of mass as the tip, then combine the vertices in the same cluster to the tip vertex
    for cluster_id in range(1, np.max(cluster_labels) + 1):
        cluster_member_ids = highest_vertex_id[cluster_labels == cluster_id]
        cluster_member_distances = vertex_to_COM_distance[cluster_member_ids]
        tip_vertex_id = cluster_member_ids[np.argmax(cluster_member_distances)]

        # assign the same label to all vertices in the same cluster
        vertex_labels[np.isin(vertex_labels, cluster_member_ids)] = tip_vertex_id

    tip_vertex_ids = np.unique(vertex_labels)

    # find the 4 tip vertices with the largest distance to center of mass
    tip_vertex_distances = vertex_to_COM_distance[tip_vertex_ids]
    top_4_tip_vertex_ids = tip_vertex_ids[np.argsort(tip_vertex_distances)[-4:]]
    top_4_tip_vertex = vertex[top_4_tip_vertex_ids, :]

    debug_plot = 0
    if debug_plot == 1: # show all the regions
        n_regions = len(tip_vertex_ids)
        cluster_colors = plt.colormaps['tab20'](np.linspace(0, 1, n_regions))
        vertex_color = np.ones((n_vertices, 4))
        for i in range(n_regions):
            vertex_color[vertex_labels == tip_vertex_ids[i], :] = cluster_colors[i]

        # 3d scatter plot with vertex colors
        fig = go.Figure(data=[
            go.Scatter3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                mode='markers',
                marker=dict(size=2, color=['rgba({},{},{},{})'.format(int(r*255), int(g*255), int(b*255), a) for r, g, b, a in vertex_color]),
            ),
        ])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show() # opens in browser

    debug_plot = 0
    if debug_plot == 1: # show the regions of the 4 tip vertices
        # build face list for Mesh3d
        fi, fj, fk = face[:, 0], face[:, 1], face[:, 2]

        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                i=fi, j=fj, k=fk,
                color='white', opacity=0.2,
                lighting=dict(ambient=1.0, diffuse=0, specular=0, roughness=1, fresnel=0),
                flatshading=True,
            ),

            # triangle edges: interleave [A, B, C, A, None] per triangle
            go.Scatter3d(
                x=np.stack([vertex[fi, 0], vertex[fj, 0], vertex[fk, 0], vertex[fi, 0], np.full(len(fi), None)], axis=1).ravel(),
                y=np.stack([vertex[fi, 1], vertex[fj, 1], vertex[fk, 1], vertex[fi, 1], np.full(len(fi), None)], axis=1).ravel(),
                z=np.stack([vertex[fi, 2], vertex[fj, 2], vertex[fk, 2], vertex[fi, 2], np.full(len(fi), None)], axis=1).ravel(),
                mode='lines',
                opacity=0.5,
                line=dict(color='gray', width=1),
            ),

            # top 4 tip vertices as large red dots
            go.Scatter3d(
                x=top_4_tip_vertex[:, 0], y=top_4_tip_vertex[:, 1], z=top_4_tip_vertex[:, 2],
                mode='markers',
                marker=dict(size=8, color='red'),
            ),

            # center of mass as a black dot
            go.Scatter3d(
                x=[center_of_mass[0]], y=[center_of_mass[1]], z=[center_of_mass[2]],
                mode='markers',
                marker=dict(size=8, color='black'),
            ),

            # lines from each tip vertex to the center of mass
            go.Scatter3d(
                x=np.stack([top_4_tip_vertex[:, 0], np.full(4, center_of_mass[0]), np.full(4, None)], axis=1).ravel(),
                y=np.stack([top_4_tip_vertex[:, 1], np.full(4, center_of_mass[1]), np.full(4, None)], axis=1).ravel(),
                z=np.stack([top_4_tip_vertex[:, 2], np.full(4, center_of_mass[2]), np.full(4, None)], axis=1).ravel(),
                mode='lines',
                line=dict(color='black', width=3),
            ),
        ])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show() # opens in browser

    return center_of_mass, top_4_tip_vertex, top_4_tip_vertex_ids, vertex_labels


