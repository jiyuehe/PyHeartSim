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

#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import modules
import numpy as np # pip install numpy
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

directory = {}
directory['home'] = script_dir
directory['result'] = script_dir.parent / 'result'

#%%
geometry_flag = 2
# 0: 2D sheet
# 1: 3D slab, regular
# 2: 3D slab, long, for computing conduction velocity
# 3: 3D hollow cube

voxel = []
neighbor_id_2d = []
voxel_for_each_vertex = []
vertex_for_each_voxel = []
vertex = []
face = []
match geometry_flag:
    case 0: # 2D sheet
        file_name = 'sheet'

        lx = 128/2 # half length, unit: mm
        ly = 128/2 # half length, unit: mm

        # create the Cartesian voxels
        xCoordinates = np.arange(-lx, lx, 1)
        yCoordinates = np.arange(-ly, ly, 1)

        nx = len(xCoordinates)
        ny = len(yCoordinates)

        id = 0
        voxel = np.zeros((nx*ny, 3))
        for i in range(nx):
            for j in range(ny):
                voxel[id, :] = [xCoordinates[i], yCoordinates[j], 0]
                id += 1
    case 1 | 2 | 3: # 3D slab
        if geometry_flag == 1: # regular slab
            file_name = 'regular_slab'

            # half lengths, unit: mm
            lx = 40/2
            ly = 50/2
            lz = 20/2
        elif geometry_flag == 2: # long slab
            file_name = 'long_slab'

            # half lengths, unit: mm
            lx = 50/2
            ly = 10/2
            lz = 10/2
        elif geometry_flag == 3: # hollow cube
            file_name = 'hollow_slab'

            # half lengths, unit: mm
            lx = 64/2
            ly = 64/2
            lz = 64/2

        # create the Cartesian voxels
        xCoordinates = np.arange(-lx, lx, 1)
        yCoordinates = np.arange(-ly, ly, 1)
        zCoordinates = np.arange(-lz, lz, 1)
        nx = len(xCoordinates)
        ny = len(yCoordinates)
        nz = len(zCoordinates)
        id = 0
        voxel = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    voxel.append([xCoordinates[i], yCoordinates[j], zCoordinates[k]])
                    id += 1
        voxel = np.array(voxel)

        if geometry_flag == 3: # hollow cube
            # remove internal voxels to create a hollow cube
            voxel_filtered = []
            for v in voxel:
                if (v[0] == max(xCoordinates) or v[0] == min(xCoordinates)) or (v[1] == max(yCoordinates) or v[1] == min(yCoordinates)) or (v[2] == max(zCoordinates) or v[2] == min(zCoordinates)):
                    voxel_filtered.append(v)
            voxel = np.array(voxel_filtered)

# for each voxel, find its neighbor voxels
neighbor_id_2d = modules.find_neighbor_voxel_ids.execute(voxel)

debug_plot = 1
if debug_plot == 1: # show geometry voxel
    n_voxel = len(voxel)
    colors = np.array(['gray'] * n_voxel)
    sizes = np.ones(n_voxel) * 3
    fig = go.Figure(data=[go.Scatter3d(
            x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
            mode='markers',
            marker=dict(size=sizes, color=colors),
            showlegend=False)])
    fig.update_layout(scene=dict(aspectmode='data')) # set aspect ratio to be equal
    fig.show()

# save geometry data
geometry = {}
geometry = {
    'Delta': 1, # voxel spacing
    'voxel': voxel, 
    'neighbor_id_2d': neighbor_id_2d,
    'voxel_for_each_vertex': voxel_for_each_vertex,
    'vertex_for_each_voxel': vertex_for_each_voxel,
    'vertex': vertex, # triangular mesh vertex
    'face': face, # triangular mesh face
}

file_path = directory['data'] / file_name
np.save(file_path, geometry)

print('done')
#%%
