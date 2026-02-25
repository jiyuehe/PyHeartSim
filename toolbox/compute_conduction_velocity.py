#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import sys
sys.path.insert(0, str(script_dir.parent))  # add PyHeartSim directory to path
import codes as parent_codes
import display_simulation_movie
import heart_sim_individual
import numpy as np # pip install numpy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

#%%
# geometry
geometry_flag = 3 # long slab for computing conduction velocity
create_geometry_flag = 0 # 0: load existing geometry, 1: create geometry
if create_geometry_flag == 0:
    geometry_data = parent_codes.geometry.process_geometry.load(geometry_flag, script_dir.parent)
elif create_geometry_flag == 1:
    geometry_data = parent_codes.geometry.process_geometry.execute(geometry_flag, script_dir.parent)

debug_plot = 0
if debug_plot == 1: # show geometry voxel
    voxel = geometry_data['voxel']
    fig = go.Figure(data=[go.Scatter3d(
            x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
            mode='markers',
            marker=dict(size=1, color='gray'),
            showlegend=False)])
    fig.update_layout(scene=dict(aspectmode='data')) # set aspect ratio to be equal
    fig.show()

# simulation parameters
simulation_parameters = {
    'geometry_flag': geometry_flag, 
    'heart_model_flag': 0, 
    # 0: Mitchell-Schaeffer model
    # 1: Aliev-Panfilov model
    'arrhythmia_flag': 0, # focal
    'compute_electrogram_flag': 0, # 0: do not compute electrogram
    'electrode_id': [], # electrode locations for computing electrograms
    't_final': 300, # ms
    'dt': 0.5, # ms. dt > 1.3 may not work well due to numerical instability
}

data_folder = script_dir.parent / 'result'

input_arguments = {}
input_arguments['geometry_data'] = geometry_data
input_arguments['simulation_parameters'] = simulation_parameters
input_arguments['s1'] = []
input_arguments['s2'] = []
input_arguments['save_result_flag'] = 1 # 1: save simulation results
input_arguments['data_folder'] = data_folder

# run simulation
heart_sim_individual.execute(input_arguments)

# display simulation movie
do_flag = 0
if do_flag == 1:
    save_movie_flag = 0 # 1: save movie. 0: do not save movie
    starting_time = 0 # 0 # ms
    ending_time = [] # ms. []: till the end. or specify a value
    sim_file_name = 'simulation_results.npz'
    data_file_path = script_dir.parent / 'result' 
    simulation_results_file_name = data_file_path / sim_file_name
    movie_save_dir = script_dir.parent / 'result' / sim_file_name.replace('.npz', '.gif') 
    simulation_results = np.load(data_file_path / simulation_results_file_name) # load simulation results
    input_arguments = {}
    input_arguments['save_movie_flag'] = save_movie_flag
    input_arguments['starting_time'] = starting_time
    input_arguments['ending_time'] = ending_time
    input_arguments['simulation_results_file_name'] = simulation_results_file_name
    input_arguments['movie_save_dir'] = movie_save_dir
    input_arguments['simulation_results'] = simulation_results
    display_simulation_movie.execute(input_arguments)

#%%
# compute conduction velocity
x_coordinates = geometry_data['voxel'][:, 0]
voxels_1 = np.where(x_coordinates == np.min(x_coordinates))[0]
voxels_2 = np.where(x_coordinates == np.max(x_coordinates))[0]

# load simulation results
sim_data = np.load(data_folder / 'simulation_results.npz')

action_potential = sim_data['action_potential']
physical_time = sim_data['physical_time']

voxel_1 = voxels_1[0]
voxel_2 = voxels_2[0]
x_1 = geometry_data['voxel'][voxel_1, 0]
x_2 = geometry_data['voxel'][voxel_2, 0]

ap_1 = action_potential[voxel_1, :]
peaks, _ = find_peaks(ap_1)
first_peak_index_1 = peaks[0]

ap_2 = action_potential[voxel_2, :]
peaks, _ = find_peaks(ap_2)
first_peak_index_2 = peaks[0]

debug_plot = 0
if debug_plot == 1: 
    plt.figure()
    plt.plot(physical_time, ap_1, label='Voxel 1')
    plt.plot(physical_time, ap_2, label='Voxel 2')
    plt.scatter(physical_time[first_peak_index_1], ap_1[first_peak_index_1], color='red', label='First Peak Voxel 1')
    plt.scatter(physical_time[first_peak_index_2], ap_2[first_peak_index_2], color='green', label='First Peak Voxel 2')
    plt.xlabel('Time (ms)')
    plt.ylabel('Action Potential (scaled)')
    plt.show()

time_diff = physical_time[first_peak_index_2] - physical_time[first_peak_index_1] # ms
distance = x_2 - x_1 # mm
conduction_velocity = distance / time_diff # mm/ms
print(f'Conduction Velocity: {conduction_velocity} mm/ms')
