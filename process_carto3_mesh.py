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

import numpy as np
import modules

directory = {}
directory['home'] = script_dir
directory['data'] = script_dir.parent / '0_data'
directory['result'] = script_dir.parent / '0_result'

name_prefix = '103_1-lagood' # nice sinus rhythm

#%%
# the original mesh
# ==============================
# original .obj mesh
vertex_original, face_original = modules.load_obj.execute(directory, name_prefix)

# manually clean the mesh in software MeshLab
# ==============================
# some useful tools in MeshLab:
# make the triangles uniform
#   Filters -> 
#       Remeshing, Simplification Reconstruction -> 
#           Simplification: Quadric Edge Collapse Decimation (Target number of faces set to 1500)
#           Close Holes
#           Smoothing, Fairing and Deformation -> Laplacian Smooth (Smoothing steps set to 1)
#           Remeshing: Isotropic Explicit Remeshing (set inter-vertex distance (Target Length) 0.5 mm)
# cut holes
# NOTE: remesh it to have edge length of 0.5mm, so that later when converting to 1mm spacing voxels, there will not have holes
#       do another remesh to have edge length of 3mm, for saving simulation data

# load the refined .obj mesh (0.5 mm resolution)
vertex, face = modules.load_obj.execute(directory, name_prefix + '_refined')

# load the 3 mm resolution .obj mesh
vertex3mm, face3mm = modules.load_obj.execute(directory, name_prefix + '_refined_3mm')

#%%
# convert triangular mesh to cartesian nodes
# ==============================
Delta = 1 # voxel spacing, unit: mm. 
# NOTE: 
# Delta = 1 is the most convenient, or grid will not be at integer values. 
# integer values make it easy for 3D convolution that is common in neural networks
thickness = 2 # how many voxels across endocardium to epicardium
voxel = modules.convert_triangular_mesh_to_cartesian_nodes.execute(vertex, face, Delta, thickness)

# for each voxel, find its neighbor voxels
neighbor_id_2d = modules.find_neighbor_voxel_ids.execute(voxel)

# id mapping between voxel and vertex3mm
voxel_id_of_vertex3mm, vertex3mm_id_of_voxel = modules.id_mapping_between_voxel_and_vertex.execute(voxel, vertex3mm)

# save geometry data
# ==============================
geometry = {}
geometry['vertex_original'] = vertex_original
geometry['face_original'] = face_original
geometry['vertex'] = vertex # high resolution mesh
geometry['face'] = face # high resolution mesh
geometry['vertex3mm'] = vertex3mm # low resolution mesh
geometry['face3mm'] = face3mm # low resolution mesh
geometry['Delta'] = Delta # voxel spacing, unit: mm
geometry['voxel'] = voxel
geometry['neighbor_id_2d'] = neighbor_id_2d # for each voxel, its neighbor voxel ids
geometry['voxel_id_of_vertex3mm'] = voxel_id_of_vertex3mm # these are voxel ids: the nearest voxel id of each vertex3mm
geometry['vertex3mm_id_of_voxel'] = vertex3mm_id_of_voxel # these are vertex3mm ids:: the nearest vertex3mm id of each voxel

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    modules.debug_plot.plot_mesh(geometry)

#%%
# save
# ==============================
file_path = directory['data'] / (name_prefix + '_geometry.npz') # save as .npz, the most compatible format for different versions of Python and Numpy
np.savez(file_path, **geometry)

print('done')
#%%
