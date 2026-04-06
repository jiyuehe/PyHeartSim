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
import geometry_processing
from scipy.spatial import cKDTree

directory = {}
directory['home'] = script_dir
# directory['data'] = script_dir / 'patient_atrium_mesh_database'
# directory['result'] = script_dir / 'result'
directory['data'] = script_dir.parent / '0_data'
directory['result'] = script_dir.parent / '0_data'

# create the folder if it does not exist
directory['result'].mkdir(exist_ok=True)

name_prefix = '103_1-lagood' # nice sinus rhythm

#%%
# the original mesh
# ==============================
# original .obj mesh
vertex_original, face_original = geometry_processing.load_obj.execute(directory['data'], name_prefix)

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
vertex, face = geometry_processing.load_obj.execute(directory['result'], name_prefix + '_refined')

# load the 3 mm resolution .obj mesh
vertex3mm, face3mm = geometry_processing.load_obj.execute(directory['result'], name_prefix + '_refined_3mm')

#%%
# convert triangular mesh to cartesian nodes for heart simulation
# ==============================
Delta = 1 # voxel spacing, unit: mm. This is a high resolution voxelization, for computing heart simulation
# NOTE: 
# Delta = 1 is the most convenient, or grid will not be at integer values
# integer values make it easy for 3D convolution that is common in neural networks
thickness = 2 # how many voxels across endocardium to epicardium
voxel = geometry_processing.convert_triangular_mesh_to_cartesian_nodes.execute(vertex, face, Delta, thickness)
neighbor_id_2d = geometry_processing.find_neighbor_voxel_ids.execute(voxel) # for each voxel, find its neighbor voxels

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel)

#%%
# create voxels for the 3mm resolution mesh, for saving simulation data
# ==============================
Delta = 3 # voxel spacing, unit: mm
thickness = 2 # how many voxels across endocardium to epicardium
voxel2 = geometry_processing.convert_triangular_mesh_to_cartesian_nodes.execute(vertex, face, Delta, thickness)

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel2)

# for each vertex, find its nearest voxel2 id
voxel2_id_of_vertex, vertex_id_of_voxel2 = geometry_processing.id_mapping_between_voxel_and_vertex.execute(voxel2, vertex)

# remove duplicates
voxel2_id_of_vertex = np.unique(voxel2_id_of_vertex)
voxel3mm = voxel2[voxel2_id_of_vertex, :]

# for each voxel3mm, find the voxel's (1mm spacing) id of the nearest voxel (1mm spacing)
tree = cKDTree(voxel)
_, voxel_id_of_voxel3mm = tree.query(voxel3mm, k=1)

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel3mm)

    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel[voxel_id_of_voxel3mm,:])

# rescale coordinates: 3mm spacing -> 1mm spacing (divide by Delta=3), so neighboring voxels are 1 unit apart, ready for use as indices
voxel3mm_1mm_spacing = np.round(voxel3mm / Delta).astype(int)

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel3mm_1mm_spacing)

#%%
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
geometry['voxel3mm'] = voxel3mm # these are voxels of 3mm spacing
geometry['voxel3mm_1mm_spacing'] = voxel3mm_1mm_spacing # these are the voxel3mm but re-scale to have 1mm spacing, so neighboring voxels are 1 unit apart, ready for use as indices
geometry['voxel_id_of_voxel3mm'] = voxel_id_of_voxel3mm # for each voxel3mm, the id of the nearest voxel (1mm spacing)

debug_plot = 0
if debug_plot == 1:
    # plot mesh and voxel
    geometry_processing.debug_plot.plot_mesh(vertex, face, voxel)

#%%
# save
# ==============================
file_path = directory['result'] / (name_prefix + '_geometry.npz') # save as .npz, the most compatible format for different versions of Python and Numpy
np.savez(file_path, **geometry)

print('done')
#%%
