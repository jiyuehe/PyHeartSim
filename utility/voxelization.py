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
import utility

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def convert(vertex, face, Delta, thickness):
    # NOTE: Delta = 1 is the most convenient. Or grid will not be at integer values. Integer values make it easy for 3D convolution that is common in neural networks

    # create a grid, so that will know what x, y, z values to take
    grid = []
    for i in range(3):
        # make the grid at integer values
        # +2*Delta is to make the grid larger, so it won't lost voxels due to discrete of Delta
        start = int(np.round(np.min(vertex[:, i]) - 2 * Delta))
        end = int(np.round(np.max(vertex[:, i]) + 2 * Delta + Delta))
    
        grid.append(np.arange(start, end, Delta))

    debug_plot = 0
    if debug_plot == 1:
        dot_size = 2
        fig = go.Figure(data=[
            # mesh
            go.Mesh3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                i=face[:, 0], j=face[:, 1], k=face[:, 2],
                color='white',
                opacity=0.8,
                showlegend=False
            ),
            # red dots (x-axis)
            go.Scatter3d(
                x=grid[0], y=np.zeros(len(grid[0])), z=np.zeros(len(grid[0])),
                mode='markers', marker=dict(size=dot_size, color='red'), showlegend=False
            ),
            # green dots (y-axis)
            go.Scatter3d(
                x=np.zeros(len(grid[1])), y=grid[1], z=np.zeros(len(grid[1])),
                mode='markers', marker=dict(size=dot_size, color='green'), showlegend=False
            ),
            # blue dots (z-axis)
            go.Scatter3d(
                x=np.zeros(len(grid[2])), y=np.zeros(len(grid[2])), z=grid[2],
                mode='markers', marker=dict(size=dot_size, color='blue'), showlegend=False
            )
        ])
        fig.show()

    # for each vertex, create voxels within the sphere of radius d_threshold
    # d_threshold = (thickness / 2) * np.sqrt(2) * Delta
    d_threshold = (thickness / 2) * Delta
    voxel_temp = []
    for n in range(vertex.shape[0]):
        if (n+1) % (vertex.shape[0] // 5) == 0:
            print(f'create voxels for each vertex {(n+1) / vertex.shape[0] * 100:.0f}%')
        
        # create a cube of points around this vertex
        sub_grid = []
        for i in range(3):
            id_mask = (grid[i] >= vertex[n, i] - d_threshold) & (grid[i] <= vertex[n, i] + d_threshold)
            sub_grid.append(grid[i][id_mask])
        
        # Create meshgrid and flatten to get all combinations
        mesh = np.meshgrid(sub_grid[0], sub_grid[1], sub_grid[2], indexing='ij')
        points = np.stack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()], axis=1)
                
        # remove points further than d_threshold
        vect = vertex[n, :] - points
        d = np.sqrt(np.sum(vect**2, axis=1))
        id_to_remove = d > d_threshold
        points = points[~id_to_remove, :]
        
        voxel_temp.append(points)

    voxel_temp = np.vstack(voxel_temp)
    voxel = np.unique(voxel_temp, axis=0)

    debug_plot = 0
    if debug_plot == 1:
        # plot mesh and voxel
        utility.debug_plot.plot_mesh(vertex, face, voxel)

    return voxel

def find_neighbor_voxel_ids(voxel):
    # neighbor_id_2d dimension is number of voxels x 18
    # neighbor_id_2d(a voxel id, the 18 neighboring voxel ids)
    
    neighbor_id_2d = np.full((voxel.shape[0], 18), -1, dtype=int)  # initialize with -1, which means no neighbor
    # 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
    # 6: +x+y, 7: -x+y, 8: +x-y, 9: -x-y
    # 10: +y+z, 11: -y+z, 12: +y-z, 13: -y-z
    # 14: +x+z, 15: -x+z, 16: +x-z, 17: -x-z
    
    # define the 18 neighbor offsets
    Delta = 1
    offsets = np.array([
        [Delta, 0, 0],      # 0: +x
        [-Delta, 0, 0],     # 1: -x
        [0, Delta, 0],      # 2: +y
        [0, -Delta, 0],     # 3: -y
        [0, 0, Delta],      # 4: +z
        [0, 0, -Delta],     # 5: -z
        [Delta, Delta, 0],  # 6: +x+y
        [-Delta, Delta, 0], # 7: -x+y
        [Delta, -Delta, 0], # 8: +x-y
        [-Delta, -Delta, 0],# 9: -x-y
        [0, Delta, Delta],  # 10: +y+z
        [0, -Delta, Delta], # 11: -y+z
        [0, Delta, -Delta], # 12: +y-z
        [0, -Delta, -Delta],# 13: -y-z
        [Delta, 0, Delta],  # 14: +x+z
        [-Delta, 0, Delta], # 15: -x+z
        [Delta, 0, -Delta], # 16: +x-z
        [-Delta, 0, -Delta] # 17: -x-z
    ])

    # build KDTree for fast neighbor search
    tree = KDTree(voxel)

    for voxel_id in range(voxel.shape[0]):
        if (voxel_id+1) % (voxel.shape[0] // 5) == 0:
            print(f'find neighbor voxels {(voxel_id+1) / voxel.shape[0] * 100:.0f}%')
        
        # calculate all neighbor positions at once
        neighbor_positions = voxel[voxel_id] + offsets
        
        # query tree for each neighbor position
        d_threshold = Delta * 0.1  # small tolerance for floating point errors
        for i, pos in enumerate(neighbor_positions):
            # find the nearest neighbor
            dist, idx = tree.query(pos)
            # only assign if the distance is very small (voxel exists at that position)
            if dist < d_threshold:
                neighbor_id_2d[voxel_id, i] = idx
    
    debug_plot = 0
    if debug_plot == 1:
        # find a voxel that has all 18 neighbors
        center_ids = np.where(np.sum(neighbor_id_2d != -1, axis=1) == 18)[0] 
        center_id = center_ids[0]
        ids = neighbor_id_2d[center_id, :]

        # show the voxel and all its 18 neighbors together
        fig = go.Figure(data=[
            # all voxels
            go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=1, color='gray'),
                showlegend=False
            ),
            # a voxel
            go.Scatter3d(
                x=[voxel[center_id, 0]], y=[voxel[center_id, 1]], z=[voxel[center_id, 2]],
                mode='markers',
                marker=dict(size=3, color='red'),
                showlegend=False
            ),
            # neighbors of the voxel
            go.Scatter3d(
                x=voxel[ids, 0], y=voxel[ids, 1], z=voxel[ids, 2],
                mode='markers',
                marker=dict(size=3, color='blue'),
                showlegend=False
            ),
        ])
        fig.show()
        
        # show each neighbor voxel to check if they are correct
        # 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
        # 6: +x+y, 7: -x+y, 8: +x-y, 9: -x-y
        # 10: +y+z, 11: -y+z, 12: +y-z, 13: -y-z
        # 14: +x+z, 15: -x+z, 16: +x-z, 17: -x-z
        for n in range(18):
            id = ids[n]
            
            fig = go.Figure(data=[
                # all voxels
                go.Scatter3d(
                    x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='gray'),
                    showlegend=False
                ),
                # a voxel
                go.Scatter3d(
                    x=[voxel[center_id, 0]], y=[voxel[center_id, 1]], z=[voxel[center_id, 2]],
                    mode='markers',
                    marker=dict(size=3, color='red'),
                    showlegend=False
                ),
                # a neighbor of the voxel
                go.Scatter3d(
                    x=[voxel[id, 0]], y=[voxel[id, 1]], z=[voxel[id, 2]],
                    mode='markers',
                    marker=dict(size=3, color='blue'),
                    showlegend=False
                ),
            ])
            fig.show()
    
    return neighbor_id_2d

def id_mapping_between_voxel_and_vertex(voxel, vertex):
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
