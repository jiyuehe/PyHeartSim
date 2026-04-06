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
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser
import geometry_processing

def execute(vertex, face, Delta, thickness):
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
        geometry_processing.debug_plot.plot_mesh(vertex, face, voxel)

    return voxel
