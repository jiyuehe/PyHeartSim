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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt # for plotting
import configuration
import utility
import common
import numpy as np # pip install numpy

#%%
directory = configuration.directory_setup()

geometry_flag = 2
# 0: 2D sheet
# 1: 3D slab, typical shape
# 2: 3D hollow cube

voxel = []
neighbor_id_2d = []
voxel_id_of_vertex = []
vertex_id_of_voxel = []
vertex = []
face = []
match geometry_flag:
    case 0: # 2D sheet
        name_prefix = 'sheet'

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
    case 1 | 2: # 3D slab
        if geometry_flag == 1: # regular slab
            name_prefix = 'regular_slab'
            # half lengths, unit: mm
            lx = 40/2
            ly = 50/2
            lz = 20/2
        elif geometry_flag == 2: # hollow cube
            name_prefix = 'hollow_cube'
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
neighbor_id_2d = utility.voxelization.find_neighbor_voxel_ids(voxel)

# voxel locations for computing unipolar electrograms
N = 2000
n_voxel = voxel.shape[0]
voxel_id_of_simulation_electrode = np.sort(np.random.choice(n_voxel, size=min(N, n_voxel), replace=False)) 

#%%
debug_plot = 1
if debug_plot == 1: 
    # show geometry voxels in gray, electrode voxels in blue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], s=0.5, c='gray', marker='.', depthshade=False, alpha=1, edgecolors='none')
    ax.scatter(voxel[voxel_id_of_simulation_electrode, 0], voxel[voxel_id_of_simulation_electrode, 1], voxel[voxel_id_of_simulation_electrode, 2], s=2, c='blue', marker='.', depthshade=False, alpha=1, edgecolors='none')
    common.set_axes_equal(ax)
    file_path = directory['data'] / (name_prefix + '_geometry.png')
    plt.savefig(file_path, dpi=300)
    plt.close()
    common.crop_image(file_path)

# save geometry data
geometry = {}
geometry = {
    'Delta': 1, # voxel spacing
    'voxel': voxel, 
    'neighbor_id_2d': neighbor_id_2d,
    'voxel_id_of_vertex': voxel_id_of_vertex,
    'vertex_id_of_voxel': vertex_id_of_voxel,
    'vertex': vertex, # triangular mesh vertex
    'face': face, # triangular mesh face
    'voxel_id_of_simulation_electrode': voxel_id_of_simulation_electrode, # these locations are used for computing unipolar electrograms
}

file_path = directory['data'] / (name_prefix + '_geometry.npz') # save as .npz, the most compatible format for different versions of Python and Numpy
np.savez(file_path, **geometry)

print('done')
#%%
