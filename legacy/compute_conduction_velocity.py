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
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utility
import simulation_individual
import configuration
import numpy as np # pip install numpy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

#%%
directory = configuration.directory_setup() # set up directories
directory['result'] = directory['home'] / 'result'
directory['result'].mkdir(exist_ok=True)

name_prefix = 'long_slab'

save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results
plot_lat_map_flag = 1 # 1: plot local activation time map. 0: do not plot local activation time map

# load geometry
file_path = directory['data'] / f'{name_prefix}_geometry.npz'
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}
n_voxel = geometry_data['voxel'].shape[0]

debug_plot = 0
if debug_plot == 1: 
    # show geometry voxel
    voxel = geometry_data['voxel']
    fig = go.Figure(data=[go.Scatter3d(
            x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
            mode='markers',
            marker=dict(size=1, color='gray'),
            showlegend=False)])
    fig.update_layout(scene=dict(aspectmode='data')) # set aspect ratio to be equal
    fig.show()

s1 = []
s2 = []

simulation_parameters, arrhythmia_parameters, heart_model_parameters = configuration.assign_simulation_parameters(name_prefix, geometry_data, s1, s2)
simulation_parameters['save_action_potential_of_all_voxel_flag'] = 1

input_arguments = {}
input_arguments['name_prefix'] = name_prefix
input_arguments['geometry_data'] = geometry_data
input_arguments['save_result_flag'] = save_result_flag
input_arguments['result_folder'] = directory['result']
input_arguments['s1'] = s1
input_arguments['s2'] = s2
input_arguments['simulation_parameters'] = simulation_parameters
input_arguments['arrhythmia_parameters'] = arrhythmia_parameters
input_arguments['heart_model_parameters'] = heart_model_parameters

# run simulation
simulation_individual.run_simulation(input_arguments)

# display simulation movie
do_flag = 1
if do_flag == 1:
    # load simulation results
    simulation_results = dict(np.load(directory['result'] / 'long_slab_simulation_results.npz', allow_pickle=False))

    save_movie_flag = 0 # 1: save movie. 0: do not save movie
    starting_time = 0 # 0 # ms
    ending_time = 1000 # ms. []: till the end. or specify a value

    simulation_results_file_name = directory['result'] / f'{name_prefix}_simulation_results.gif'
    movie_save_dir = directory['result'] / simulation_results_file_name

    in_arg = {}
    in_arg['save_movie_flag'] = save_movie_flag
    in_arg['starting_time'] = starting_time
    in_arg['ending_time'] = ending_time
    in_arg['simulation_results_file_name'] = simulation_results_file_name
    in_arg['movie_save_dir'] = movie_save_dir
    in_arg['simulation_results'] = simulation_results
    in_arg['geometry_data'] = geometry_data
    in_arg['save_action_potential_of_all_voxel_flag'] = simulation_parameters['save_action_potential_of_all_voxel_flag']
    utility.display_simulation_movie.execute(in_arg)

#%%
# typical human left atrium conduction velocity is 0.5 to 1.0 mm/ms

# compute conduction velocity, method 1
x_coordinates = geometry_data['voxel'][:, 0]
voxels_1 = np.where(x_coordinates == np.min(x_coordinates))[0]
voxels_2 = np.where(x_coordinates == np.max(x_coordinates))[0]

# load simulation results
simulation_results = dict(np.load(directory['result'] / 'long_slab_simulation_results.npz', allow_pickle=False))

action_potential = simulation_results['action_potential']
physical_time = simulation_results['physical_time']

voxel_1 = voxels_1[0]
voxel_2 = voxels_2[0]
x_1 = geometry_data['voxel'][voxel_1, 0]
x_2 = geometry_data['voxel'][voxel_2, 0]

ap_1 = action_potential[:, voxel_1]
peaks, _ = find_peaks(ap_1)
first_peak_index_1 = peaks[0]

ap_2 = action_potential[:, voxel_2]
peaks, _ = find_peaks(ap_2)
first_peak_index_2 = peaks[0]

debug_plot = 1
if debug_plot == 1: 
    plt.figure()
    plt.plot(physical_time, ap_1, label='Voxel 1')
    plt.plot(physical_time, ap_2, label='Voxel 2')
    plt.scatter(physical_time[first_peak_index_1], ap_1[first_peak_index_1], color='red', label='First Peak Voxel 1')
    plt.scatter(physical_time[first_peak_index_2], ap_2[first_peak_index_2], color='green', label='First Peak Voxel 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Action Potential (scaled)')
    plt.legend()
    plt.savefig(directory['result'] / 'conduction_velocity_ap.png', dpi=300)
    plt.close()

time_diff = physical_time[first_peak_index_2] - physical_time[first_peak_index_1] # ms
distance = x_2 - x_1 # mm
conduction_velocity = distance / time_diff # mm/ms
print(f'conduction velocity (method 1): {conduction_velocity:.2f} mm/ms')

#%%
# compute conduction velocity, method 2
# compute local activation time
electrogram_unipolar = simulation_results['electrogram_unipolar']
lat_electrode = utility.lat_map.compute_electrode_lat(electrogram_unipolar)

voxel_id_of_simulation_electrode = geometry_data['voxel_id_of_simulation_electrode']
xyz = geometry_data['voxel'][voxel_id_of_simulation_electrode, :] # coordinates of electrode voxels
lat = lat_electrode

# compute conduction velocity for each point
from scipy.spatial import cKDTree
tree = cKDTree(xyz)
n_points = xyz.shape[0]
conduction_velocity_vectors = np.zeros((n_points, 3))
conduction_velocity_magnitudes = np.zeros(n_points)
neighbor_radius = 8.0 # mm

for i in range(n_points):
    # find neighbors (including self)
    idx = tree.query_ball_point(xyz[i], neighbor_radius)
    if len(idx) < 4:
        # not enough neighbors for 3D fit
        conduction_velocity_vectors[i, :] = np.nan
        conduction_velocity_magnitudes[i] = np.nan
        continue
    
    pts = xyz[idx]
    lats = lat[idx]

    # least-squares fit: lat = a*x + b*y + c*z + d
    A = np.column_stack((pts, np.ones(len(pts))))
    coeffs, _, _, _ = np.linalg.lstsq(A, lats, rcond=None)
    grad = coeffs[:3]  # gradient of activation time

    # conduction velocity vector: v = grad_t / |grad_t|^2
    grad_norm_sq = np.dot(grad, grad)
    if grad_norm_sq > 1e-8:
        v_vec = grad / grad_norm_sq
        v_mag = 1.0 / np.linalg.norm(grad)
        conduction_velocity_vectors[i, :] = v_vec
        conduction_velocity_magnitudes[i] = v_mag
    else:
        conduction_velocity_vectors[i, :] = np.nan
        conduction_velocity_magnitudes[i] = np.nan

# mean conduction velocity
mean_cv = np.nanmean(conduction_velocity_magnitudes)
print(f'conduction velocity (method 2): {mean_cv:.3f} mm/ms')

print('done')