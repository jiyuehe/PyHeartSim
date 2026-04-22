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
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import utility
import common

import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

#%%
# directory folder
directory = {}
directory['home'] = script_dir
directory['mesh_database'] = script_dir / 'mesh_database'

# grab all atrium mesh file names
mesh_files = list(Path(directory['mesh_database']/'left_atrium').glob('*.obj'))
name_prefixes = [mesh_file.stem for mesh_file in mesh_files]
name_prefixes_left_atrium = sorted(name_prefixes, key=lambda x: int(x.split('_')[0])) # sort by the number before the underscore

mesh_files = list(Path(directory['mesh_database']/'right_atrium').glob('*.obj'))
name_prefixes = [mesh_file.stem for mesh_file in mesh_files]
name_prefixes_right_atrium = sorted(name_prefixes, key=lambda x: int(x.split('_')[0])) # sort by the number before the underscore

do_flag = 0
if do_flag == 1: # load the mesh and save as figure
    for name_prefix in name_prefixes_left_atrium:
        print(f'processing {name_prefix}')

        vertex, face = common.load_obj(directory['mesh_database']/'left_atrium', name_prefix)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(
            vertex[face], alpha=0.5, facecolor="white", edgecolor="gray", linewidth=0.1
        )
        ax.add_collection3d(poly)
        ax.view_init(elev=70, azim=-70)
        ax.set_axis_off()
        common.set_axes_equal(ax)

        png_path = str(directory['mesh_database']/'left_atrium' / f'{name_prefix}.png')
        plt.savefig(png_path, dpi=100)
        plt.close(fig)

        common.crop_image(png_path)

    for name_prefix in name_prefixes_right_atrium:
        print(f'processing {name_prefix}')

        vertex, face = common.load_obj(directory['mesh_database']/'right_atrium', name_prefix)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(
            vertex[face], alpha=0.5, facecolor="white", edgecolor="gray", linewidth=0.1
        )
        ax.add_collection3d(poly)
        ax.view_init(elev=70, azim=-70)
        ax.set_axis_off()
        common.set_axes_equal(ax)

        png_path = str(directory['mesh_database']/'right_atrium' / f'{name_prefix}.png')
        plt.savefig(png_path, dpi=100)
        plt.close(fig)

        common.crop_image(png_path)

#%%
mesh_directory = directory['mesh_database'] / 'left_atrium'
result_directory = Path('/home/j/Desktop/hdd/share_folder/patient_data')

for n in [98]:#range(len(name_prefixes_left_atrium)):
    name_prefix = name_prefixes_left_atrium[n]
    print(f'processing {name_prefix}')

    # automatically refine the mesh
    input_mesh_path = mesh_directory / f'{name_prefix}.obj'
    output_mesh_path = result_directory / f'{name_prefix}_refined.obj'
    
    # NOTE:
    # key parameter for mesh refinement: tsdf_target_rest
    # - it determines the smoothing and hole filling 
    # - larger value -> high resolution preserves more details but might not fill holes 
    # - smaller value -> more smoothing and hole filling, but might lose details
    utility.automatic_mesh_refinement.clean_mesh(
        str(input_mesh_path),
        str(output_mesh_path),
        debug_mode = False,
        tsdf_target_res = 100,
        tsdf_truncation_dist = None,
        morph_closing_iters = 3,
        morph_opening_iters = 0,
        morph_dilation_iters = 1,
        pad_voxels = 2,
        fill_internal_volume = True,
        sdf_smoothing_sigma = 2.0,
        mc_level = 0.0,
        simplify_faces_ratio = 0.9,
        enable_decimation = True,
        smooth_iterations = 3,
        smooth_lambda = 0.6,
        enable_remesh = True,
        target_edge_length = 0.5,
        post_remesh_smooth_iterations = 2,
        visualize = False,
    )

    debug_plot = 1
    if debug_plot == 1:
        vertex_original, face_original = common.load_obj(mesh_directory, name_prefix)
        vertex_refined, face_refined = common.load_obj(result_directory, name_prefix + '_refined')

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

        for ax, verts, faces, title in [
            (axes[0], vertex_original, face_original, "original"),
            (axes[1], vertex_refined, face_refined, "processed"),
        ]:
            poly = Poly3DCollection(
                verts[faces], alpha=0.5, facecolor="white", edgecolor="gray", linewidth=0.1
            )
            ax.add_collection3d(poly)
            ax.view_init(elev=70, azim=-70)
            ax.set_axis_off()
            common.set_axes_equal(ax)
        
        png_path = str(result_directory / f'{name_prefix}_original_and_refined.png')
        plt.savefig(png_path, dpi=300)
        plt.close(fig)

        common.crop_image(png_path)

# NOTE: 
# can also use software Meshlab to manually refine the mesh
# some useful tools in MeshLab:
# Filters -> 
#     Remeshing, Simplification Reconstruction -> 
#         Simplification: Quadric Edge Collapse Decimation (Target number of faces set to 1500)
#         Close Holes
#         Smoothing, Fairing and Deformation ->
#             -> Laplacian Smooth (Smoothing steps set to 1)
#         Remeshing: Isotropic Explicit Remeshing (Target Length (inter-vertex distance) set to 0.5 mm)

#%%
# cut holes: cut the mitral valve, pulmonary veins, etc
# NOTE:
# use software Meshlab to manually cut holes in the mesh {name_prefix}_refined.obj
# save the cut mesh as {name_prefix}_refined_cut.obj

#%%
# # load the refined and holes cut .obj mesh
# vertex, face = common.load_obj(result_directory, name_prefix + '_refined_cut')

# for name_prefix in name_prefixes_left_atrium:
#     print(f'processing {name_prefix}')
#     #%%
#     # convert triangular mesh to cartesian nodes for heart simulation
#     # ==============================
#     Delta = 1 # voxel spacing, unit: mm. This is a high resolution voxelization, for computing heart simulation
#     # NOTE: 
#     # Delta = 1 is the most convenient, or grid will not be at integer values. integer values make it easy for 3D convolution that is common in neural networks
#     thickness = 2 # how many voxels across endocardium to epicardium
#     voxel = utility.voxelization.convert(vertex, face, Delta, thickness)
#     neighbor_id_2d = utility.voxelization.find_neighbor_voxel_ids(voxel) # for each voxel, find its neighbor voxels

#     debug_plot = 0
#     if debug_plot == 1:
#         # plot mesh and voxel
#         utility.debug_plot.plot_mesh(vertex, face, voxel)

#     #%%
#     # create voxels for the 3mm resolution mesh, for saving simulation data
#     # ==============================
#     Delta = 3 # voxel spacing, unit: mm
#     thickness = 2 # how many voxels across endocardium to epicardium
#     voxel2 = utility.voxelization.convert(vertex, face, Delta, thickness)

#     debug_plot = 0
#     if debug_plot == 1:
#         # plot mesh and voxel
#         utility.debug_plot.plot_mesh(vertex, face, voxel2)

#     # for each vertex, find its nearest voxel2 id
#     voxel2_id_of_vertex, vertex_id_of_voxel2 = utility.voxelization.id_mapping_between_voxel_and_vertex(voxel2, vertex)

#     # remove duplicates
#     voxel2_id_of_vertex = np.unique(voxel2_id_of_vertex)
#     voxel3mm = voxel2[voxel2_id_of_vertex, :]

#     # for each voxel3mm, find the voxel's (1mm spacing) id of the nearest voxel (1mm spacing)
#     tree = cKDTree(voxel)
#     _, voxel_id_of_voxel3mm = tree.query(voxel3mm, k=1)

#     debug_plot = 0
#     if debug_plot == 1:
#         # plot mesh and voxel
#         utility.debug_plot.plot_mesh(vertex, face, voxel3mm)

#         utility.debug_plot.plot_mesh(vertex, face, voxel[voxel_id_of_voxel3mm,:])

#     # rescale coordinates: 3mm spacing -> 1mm spacing (divide by Delta=3), so neighboring voxels are 1 unit apart, ready for use as indices
#     voxel3mm_1mm_spacing = np.round(voxel3mm / Delta).astype(int)

#     debug_plot = 0
#     if debug_plot == 1:
#         # plot mesh and voxel
#         utility.debug_plot.plot_mesh(vertex, face, voxel3mm_1mm_spacing)

#     #%%
#     # save geometry data
#     # ==============================
#     geometry = {}
#     geometry['vertex_original'] = vertex_original
#     geometry['face_original'] = face_original
#     geometry['vertex'] = vertex # high resolution mesh
#     geometry['face'] = face # high resolution mesh
#     geometry['Delta'] = Delta # voxel spacing, unit: mm
#     geometry['voxel'] = voxel
#     geometry['neighbor_id_2d'] = neighbor_id_2d # for each voxel, its neighbor voxel ids
#     geometry['voxel3mm'] = voxel3mm # coordinates: these are voxels of 3mm spacing
#     geometry['voxel3mm_1mm_spacing'] = voxel3mm_1mm_spacing # coordinates: these are the voxel3mm but re-scale to have 1mm spacing, so neighboring voxels are 1 unit apart, ready for use as indices
#     geometry['voxel_id_of_simulation_electrode'] = voxel_id_of_voxel3mm # voxel ids: for each voxel3mm, the id of the nearest voxel (1mm spacing)

#     #%%
#     # save
#     # ==============================
#     file_path = directory['data'] / (name_prefix + '_geometry.npz') # save as .npz, the most compatible format for different versions of Python and Numpy
#     np.savez(file_path, **geometry)

print('done')
#%%
