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

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

import numpy as np

def add_voxel_cubes(fig, voxel_arr):
    # each voxel is a center, create a cube of 1mm (0.5mm offset each direction)
    cube_vertices = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ])
    # faces of the cube (each face is two triangles)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], # bottom
        [4, 5, 6], [4, 6, 7], # top
        [0, 1, 5], [0, 5, 4], # front
        [2, 3, 7], [2, 7, 6], # back
        [1, 2, 6], [1, 6, 5], # right
        [3, 0, 4], [3, 4, 7] # left
    ])
    # for performance, combine all cubes into one Mesh3d
    all_vertices = []
    all_i, all_j, all_k = [], [], []
    for idx, center in enumerate(voxel_arr):
        verts = center + cube_vertices
        base = 8 * idx
        all_vertices.append(verts)
        for f in faces:
            all_i.append(base + f[0])
            all_j.append(base + f[1])
            all_k.append(base + f[2])
    all_vertices = np.vstack(all_vertices)
    fig.add_trace(go.Mesh3d(
        x=all_vertices[:, 0],
        y=all_vertices[:, 1],
        z=all_vertices[:, 2],
        i=all_i,
        j=all_j,
        k=all_k,
        color='grey',
        opacity=1,
        name='Voxels',
        showlegend=True
    ))

def plot_mesh(vertex, face, voxel):
    fig = go.Figure()
    # add mesh faces
    fig.add_trace(go.Mesh3d(
        x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
        i=face[:, 0], j=face[:, 1], k=face[:, 2],
        opacity=0.3,
        flatshading=True,
        color='lightgray',
        showscale=False,
        hoverinfo='skip'
    ))

    # add triangle edges
    edges = np.vstack([face[:, [0, 1]], face[:, [1, 2]], face[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    edge_x, edge_y, edge_z = [], [], []
    for e in edges:
        p0 = vertex[e[0]]
        p1 = vertex[e[1]]
        edge_x.extend([p0[0], p1[0], None])
        edge_y.extend([p0[1], p1[1], None])
        edge_z.extend([p0[2], p1[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='skip',
        showlegend=False
    ))

    # plot voxels as 1mm cubes if available and non-empty
    if len(voxel) > 0:
        add_voxel_cubes(fig, voxel)

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.show()
