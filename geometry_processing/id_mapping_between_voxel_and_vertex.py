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
from scipy.spatial import KDTree
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def execute(voxel, vertex):
    # the nearest voxel id of each vertex, len(voxel_for_each_vertex) = n_vertex, max(voxel_for_each_vertex) = n_voxel
    # ------------------------------
    N = vertex.shape[0]
    voxel_for_each_vertex = np.zeros(N, dtype=int)

    # build KDTree for fast nearest neighbor search
    voxel_tree = KDTree(voxel)

    for n in range(N):
        if (n + 1) % (N // 5) == 0:
            print(f'find the nearest voxel id of each vertex {(n + 1) / N * 100:.0f}%')
        xyz = vertex[n, :]
        
        # find nearest voxel using KDTree
        distance, id = voxel_tree.query(xyz, k=1)
        voxel_for_each_vertex[n] = id

    debug_plot = 0
    if debug_plot == 1:
        vertex_id = 1
        voxel_id = voxel_for_each_vertex[vertex_id]

        fig = go.Figure(data=[
            # all vertices
            go.Scatter3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                mode='markers',
                marker=dict(size=1, color='gray'),
                showlegend=False
            ),
            # a vertex
            go.Scatter3d(
                x=[vertex[vertex_id, 0]], y=[vertex[vertex_id, 1]], z=[vertex[vertex_id, 2]],
                mode='markers',
                marker=dict(size=3, color='red'),
                showlegend=False
            ),
            # the voxel
            go.Scatter3d(
                x=[voxel[voxel_id, 0]], y=[voxel[voxel_id, 1]], z=[voxel[voxel_id, 2]],
                mode='markers',
                marker=dict(size=3, color='blue'),
                showlegend=False
            ),
        ])
        fig.show()

    # the nearest vertex id of each voxel, len(vertex_for_each_voxel) = n_voxel, max(vertex_for_each_voxel) = n_vertex
    # ------------------------------
    N = voxel.shape[0]
    vertex_for_each_voxel = np.zeros(N, dtype=int)

    # build KDTree for fast nearest neighbor search
    vertex_tree = KDTree(vertex)

    for n in range(N):
        if (n + 1) % (N // 5) == 0:
            print(f'find the nearest vertex id of each voxel {(n + 1) / N * 100:.0f}%')
        
        xyz = voxel[n, :]
        
        # find nearest vertex using KDTree
        distance, id = vertex_tree.query(xyz, k=1)
        vertex_for_each_voxel[n] = id

    debug_plot = 0
    if debug_plot == 1:
        voxel_id = 10000
        vertex_id = vertex_for_each_voxel[voxel_id]

        fig = go.Figure(data=[
            # all vertices
            go.Scatter3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                mode='markers',
                marker=dict(size=1, color='gray'),
                showlegend=False
            ),
            # a vertex
            go.Scatter3d(
                x=[vertex[vertex_id, 0]], y=[vertex[vertex_id, 1]], z=[vertex[vertex_id, 2]],
                mode='markers',
                marker=dict(size=3, color='red'),
                showlegend=False
            ),
            # the voxel
            go.Scatter3d(
                x=[voxel[voxel_id, 0]], y=[voxel[voxel_id, 1]], z=[voxel[voxel_id, 2]],
                mode='markers',
                marker=dict(size=3, color='blue'),
                showlegend=False
            ),
        ])
        fig.show()

    return voxel_for_each_vertex, vertex_for_each_voxel
