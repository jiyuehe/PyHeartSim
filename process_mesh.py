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
import geometry_processing
import utility
import configuration

directory = configuration.directory_setup()
name_prefix = '102_1-LA FAM1' # '103_1-lagood' # atrial mesh .obj file name

#%%
# the original mesh
# ==============================
# original .obj mesh
vertex_original, face_original = utility.common.load_obj(directory['mesh_database'], name_prefix)

# automatically refine the mesh
# ==============================
input_mesh_path = directory['mesh_database'] / f'{name_prefix}.obj'
output_mesh_path = directory['result'] / f'{name_prefix}_refined.obj'

# parameter setup
debug_mode = True
tsdf_target_res = 120
tsdf_truncation_dist = None
morph_closing_iters = 3
morph_dilation_iters = 1
pad_voxels = 2
fill_internal_volume = True
sdf_smoothing_sigma = 2.0
mc_level = 0.0
simplify_faces_ratio = 0.9
enable_decimation = True
smooth_iterations = 1
smooth_lambda = 0.6
enable_remesh = True
post_remesh_smooth_iterations = 5
visualize = False

report_05mm = clean_mesh(
    str(input_mesh_path),
    str(output_mesh_path),
    debug_mode=debug_mode,
    tsdf_target_res=tsdf_target_res,
    tsdf_truncation_dist=tsdf_truncation_dist,
    morph_closing_iters=morph_closing_iters,
    morph_dilation_iters=morph_dilation_iters,
    pad_voxels=pad_voxels,
    fill_internal_volume=fill_internal_volume,
    sdf_smoothing_sigma=sdf_smoothing_sigma,
    mc_level=mc_level,
    simplify_faces_ratio=simplify_faces_ratio,
    enable_decimation=enable_decimation,
    smooth_iterations=smooth_iterations,
    smooth_lambda=smooth_lambda,
    enable_remesh=enable_remesh,
    target_edge_length=0.5,
    post_remesh_smooth_iterations=post_remesh_smooth_iterations,
    visualize=visualize,
)
print("TSDF-style rebuild finished (0.5 mm). Report:")
for key, value in report_05mm.items():
    print(f"{key}: {value}")
print()

debug_plot = 1
if debug_plot == 1:
    original_mesh = trimesh.load(str(input_mesh_path), force="mesh")
    processed_mesh = trimesh.load(str(output_mesh_path), force="mesh")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})

    for ax, mesh, title in zip(
        axes,
        [original_mesh, processed_mesh],
        ["Original Mesh", "Processed Mesh"],
    ):
        verts = mesh.vertices
        faces = mesh.faces
        poly = Poly3DCollection(
            verts[faces], alpha=0.5, facecolor="white", edgecolor="gray", linewidth=0.1
        )
        ax.add_collection3d(poly)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=70, azim=-70)
        common.set_axes_equal.execute(ax)

    # plt.tight_layout()
    
    png_path = str(directory['result'] / f'{name_prefix}_mesh_comparison.png')
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

    common.crop_image.execute(png_path)

print('done')

# NOTE: 
# can also use software Meshlab to manually refine the mesh
# some useful tools in MeshLab:
# - make the triangles uniform
#   Filters -> 
#       Remeshing, Simplification Reconstruction -> 
#           Simplification: Quadric Edge Collapse Decimation (Target number of faces set to 1500)
#           Close Holes
#           Smoothing, Fairing and Deformation ->
#               -> Laplacian Smooth (Smoothing steps set to 1)
#           Remeshing: Isotropic Explicit Remeshing (Target Length (inter-vertex distance) set to 0.5 mm)
# - cut holes: cut the mitral valve, pulmonary veins, etc.

# load the refined .obj mesh (0.5 mm resolution)
vertex, face = utility.common.load_obj(directory['result'], name_prefix + '_refined')


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
# geometry['vertex3mm'] = vertex3mm # low resolution mesh
# geometry['face3mm'] = face3mm # low resolution mesh
geometry['Delta'] = Delta # voxel spacing, unit: mm
geometry['voxel'] = voxel
geometry['neighbor_id_2d'] = neighbor_id_2d # for each voxel, its neighbor voxel ids
geometry['voxel3mm'] = voxel3mm # coordinates: these are voxels of 3mm spacing
geometry['voxel3mm_1mm_spacing'] = voxel3mm_1mm_spacing # coordinates: these are the voxel3mm but re-scale to have 1mm spacing, so neighboring voxels are 1 unit apart, ready for use as indices
geometry['voxel_id_of_voxel3mm'] = voxel_id_of_voxel3mm # voxel ids: for each voxel3mm, the id of the nearest voxel (1mm spacing)

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
