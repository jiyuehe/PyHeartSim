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

def execute(voxel):
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

'''
    # the method below is easy to understand, but runs slow
    for voxel_id in range(voxel.shape[0]):
        if voxel_id % round(voxel.shape[0] / 10) == 0:
            print(f'find neighbor voxels {voxel_id / voxel.shape[0] * 100:.1f}%')
        
        this_x = voxel[voxel_id, 0]
        this_y = voxel[voxel_id, 1]
        this_z = voxel[voxel_id, 2]
        
        # ---------------------------------------------------------------------
        # 0: +x
        n = this_x + Delta
        id_mask = (np.abs(voxel[:, 0] - n) < d_threshold) & (voxel[:, 1] == this_y) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        # cannot use ==: id = find(voxel(:,1)==n & voxel(:,2)==this_y & voxel(:,3)==this_z). because there are computer decimal errors
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 0] = ids[0]
        
        # 1: -x
        n = this_x - Delta
        id_mask = (np.abs(voxel[:, 0] - n) < d_threshold) & (voxel[:, 1] == this_y) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 1] = ids[0]
        
        # 2: +y
        n = this_y + Delta
        id_mask = (voxel[:, 0] == this_x) & (np.abs(voxel[:, 1] - n) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 2] = ids[0]
        
        # 3: -y
        n = this_y - Delta
        id_mask = (voxel[:, 0] == this_x) & (np.abs(voxel[:, 1] - n) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 3] = ids[0]
        
        # 4: +z
        n = this_z + Delta
        id_mask = (voxel[:, 0] == this_x) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 4] = ids[0]
        
        # 5: -z
        n = this_z - Delta
        id_mask = (voxel[:, 0] == this_x) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 5] = ids[0]
        
        # ---------------------------------------------------------------------
        # 6: +x+y
        n1 = this_x + Delta
        n2 = this_y + Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (np.abs(voxel[:, 1] - n2) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 6] = ids[0]
        
        # 7: -x+y
        n1 = this_x - Delta
        n2 = this_y + Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (np.abs(voxel[:, 1] - n2) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 7] = ids[0]
        
        # 8: +x-y
        n1 = this_x + Delta
        n2 = this_y - Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (np.abs(voxel[:, 1] - n2) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 8] = ids[0]
        
        # 9: -x-y
        n1 = this_x - Delta
        n2 = this_y - Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (np.abs(voxel[:, 1] - n2) < d_threshold) & (voxel[:, 2] == this_z)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 9] = ids[0]
        
        # ---------------------------------------------------------------------
        # 10: +y+z
        n1 = this_y + Delta
        n2 = this_z + Delta
        id_mask = (voxel[:, 0] == this_x) & (np.abs(voxel[:, 1] - n1) < d_threshold) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 10] = ids[0]
        
        # 11: -y+z
        n1 = this_y - Delta
        n2 = this_z + Delta
        id_mask = (voxel[:, 0] == this_x) & (voxel[:, 1] == n1) & (voxel[:, 2] == n2)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 11] = ids[0]
        
        # 12: +y-z
        n1 = this_y + Delta
        n2 = this_z - Delta
        id_mask = (voxel[:, 0] == this_x) & (np.abs(voxel[:, 1] - n1) < d_threshold) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 12] = ids[0]
        
        # 13: -y-z
        n1 = this_y - Delta
        n2 = this_z - Delta
        id_mask = (voxel[:, 0] == this_x) & (np.abs(voxel[:, 1] - n1) < d_threshold) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 13] = ids[0]
        
        # ---------------------------------------------------------------------
        # 14: +x+z
        n1 = this_x + Delta
        n2 = this_z + Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 14] = ids[0]
        
        # 15: -x+z
        n1 = this_x - Delta
        n2 = this_z + Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 15] = ids[0]
        
        # 16: +x-z
        n1 = this_x + Delta
        n2 = this_z - Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 16] = ids[0]
        
        # 17: -x-z
        n1 = this_x - Delta
        n2 = this_z - Delta
        id_mask = (np.abs(voxel[:, 0] - n1) < d_threshold) & (voxel[:, 1] == this_y) & (np.abs(voxel[:, 2] - n2) < d_threshold)
        ids = np.where(id_mask)[0]
        if len(ids) > 0:
            neighbor_id_2d[voxel_id, 17] = ids[0]

'''