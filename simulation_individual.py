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
import time
import numpy as np # pip install numpy
from numba import cuda # pip install numba
import matplotlib.pyplot as plt # pip install matplotlib
import simulation
import utility
import configuration

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

#%%
def run_simulation(input_arguments):
    s1 = input_arguments['s1']
    s2 = input_arguments['s2']
    save_result_flag = input_arguments['save_result_flag']
    result_folder = input_arguments['result_folder']
    geometry_data = input_arguments['geometry_data']
    simulation_parameters = input_arguments['simulation_parameters']
    arrhythmia_parameters = input_arguments['arrhythmia_parameters']
    heart_model_parameters = input_arguments['heart_model_parameters']

    n_voxel = geometry_data['voxel'].shape[0]

    if simulation_parameters['geometry_flag'] == 2: # long slab for computing conduction velocity
        x_coordinates = geometry_data['voxel'][:, 0]
        voxels_1 = np.where(x_coordinates == np.min(x_coordinates))[0]
        arrhythmia_parameters['s1_pacing_voxel_id'] = voxels_1

    # fiber orientations
    D0 = simulation.equation_parts.load_fiber(n_voxel)

    # heart model equation parts
    P_2d = simulation.equation_parts.heart_model_equation_parts(simulation_parameters, n_voxel, D0, geometry_data['neighbor_id_2d'], heart_model_parameters)

    # solve differential equations
    start = time.time()
    if cuda.is_available(): # GPU parallel
        print("GPU parallel computing for simulation")
        action_potential, h, physical_time = simulation.simulation_gpu.compute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters)
    end = time.time()
    print(f'simulation completed in {end - start:.1f} seconds')

    # compute unipolar electrogram
    start = time.time()
    if simulation_parameters['compute_electrogram_flag'] == 1:
        voxel = geometry_data['voxel']
        voxel_id_of_electrode = simulation_parameters['voxel_id_of_electrode']
        electrode_xyz = voxel[voxel_id_of_electrode, :]
        Delta = geometry_data['Delta']
        neighbor_id_2d = geometry_data['neighbor_id_2d']
        
        # check GPU availability and choose appropriate module
        if cuda.is_available():
            print("GPU parallel computing for electrogram")
            electrogram_unipolar = simulation.unipolar_electrogram_gpu.compute(electrode_xyz, voxel, D0, heart_model_parameters['c_voxel'], action_potential, Delta, neighbor_id_2d)
    end = time.time()
    print(f'electrogram computation completed in {end - start:.1f} seconds')

    # save simulation results
    if save_result_flag == 1:
        voxel_id_of_electrode = geometry_data['voxel_id_of_electrode']

        simulation_results = {}
        simulation_results['action_potential_electrode'] = action_potential[:, voxel_id_of_electrode] # shape: (time, n_electrode)
        simulation_results['h_electrode'] = h[:, voxel_id_of_electrode] # shape: (time, n_electrode)
        simulation_results['physical_time'] = physical_time
        simulation_results['geometry_flag'] = simulation_parameters['geometry_flag']

        if simulation_parameters['compute_electrogram_flag'] == 1:
            simulation_results['voxel_id_of_electrode'] = simulation_parameters['voxel_id_of_electrode']
            simulation_results['electrogram_unipolar'] = electrogram_unipolar

        if simulation_parameters['save_action_potential_of_all_voxel_flag'] == 1:
            simulation_results['action_potential'] = action_potential # shape: (time, n_voxel)
            simulation_results['h'] = h # shape: (time, n_voxel)

        # save simulation results
        name_prefix = input_arguments['name_prefix']
        if str(s1) != '[]' and str(s2) == '[]':
            np.savez(result_folder / f'{name_prefix}_simulation_results_{str(s1)}', **simulation_results)
        elif str(s1) == '[]' and str(s2) == '[]':
            np.savez(result_folder / f'{name_prefix}_simulation_results', **simulation_results)
        elif str(s1) != '[]' and str(s2) != '[]':
            np.savez(result_folder / f'{name_prefix}_simulation_results_{str(s1)}_{str(s2[0])}', **simulation_results)

#%%
# If running this script directly, the following code block will be executed. 
# If calling the run_simulation() function from another script, the following code block will be ignored.
if __name__ == "__main__":
    directory = configuration.directory_setup() # set up directories
    name_prefix = configuration.mesh_name() # get mesh name prefix

    save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results
    plot_lat_map_flag = 1 # 1: plot local activation time map. 0: do not plot local activation time map

    # load geometry data
    file_path = directory['data'] / f'{name_prefix}_geometry.npz'
    data = np.load(file_path, allow_pickle=False)
    geometry_data = {k: data[k] for k in data.files}
    n_voxel = geometry_data['voxel'].shape[0]

    s1 = 18591 # s1 pacing voxel id
    s2 = [] # if simulate rotor, s2 will be automatically determined by the code
    
    simulation_parameters, arrhythmia_parameters, heart_model_parameters = configuration.assign_simulation_parameters(name_prefix, geometry_data, s1, s2, n_voxel)

    s2 = arrhythmia_parameters['s2_pacing_voxel_id']

    debug_plot = 0
    if debug_plot == 1: 
        # show pacing voxels
        voxel = geometry_data['voxel']
        traces = [
            go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='voxels'
            ),
            go.Scatter3d(
                x=[voxel[s1, 0]], y=[voxel[s1, 1]], z=[voxel[s1, 2]],
                mode='markers',
                marker=dict(size=6, color='blue'),
                name='s1'
            ),
        ]
        if str(s2) != '[]':
            traces.append(go.Scatter3d(
                x=voxel[s2, 0], y=voxel[s2, 1], z=voxel[s2, 2],
                mode='markers',
                marker=dict(size=6, color='red'),
                name='s2'
            ))
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
            legend=dict(itemsizing='constant'),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.show()

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
    run_simulation(input_arguments)

    # compute local activation time
    if str(s2) == '[]':
        file_name = f'{name_prefix}_simulation_results_{s1}.npz'
    else: 
        file_name = f'{name_prefix}_simulation_results_{s1}_{s2[0]}.npz'
    simulation_results = dict(np.load(directory['result'] / file_name, allow_pickle=False)) # load simulation results
    electrogram_unipolar = simulation_results['electrogram_unipolar']
    lat_electrode = utility.lat_map.compute_electrode_lat(electrogram_unipolar)

    # interpolate local activation time from electrode locations to all voxels
    voxel = geometry_data['voxel']
    electrode_voxel = geometry_data['voxel'][geometry_data['voxel_id_of_electrode'], :]
    lat_voxel = utility.lat_map.interpolate_lat(voxel, electrode_voxel, lat_electrode)

    # plot local activation time map
    if plot_lat_map_flag == 1:
        if str(s2) == '[]':
            fig_name = directory['result'] / f'{name_prefix}_lat_{str(s1)}.png'
        else:
            fig_name = directory['result'] / f'{name_prefix}_lat_{str(s1)}_{str(s2[0])}.png'
        
        geometry_flag = simulation_results['geometry_flag']
        utility.lat_map.plot(voxel, lat_voxel, geometry_flag, fig_name)
        utility.common.crop_image(fig_name)

    # save lat to simulation_results
    simulation_results['lat_electrode'] = lat_electrode
    np.savez(directory['result'] / file_name, **simulation_results)

    # plot some action potentials and electrograms
    do_flag = 1
    if do_flag == 1: 
        # load simulation results
        if str(s2) == '[]':
            simulation_results = dict(np.load(directory['result'] / f'{name_prefix}_simulation_results_{str(s1)}.npz', allow_pickle=False))
        else:
            simulation_results = dict(np.load(directory['result'] / f'{name_prefix}_simulation_results_{str(s1)}_{str(s2[0])}.npz', allow_pickle=False))

        action_potential = simulation_results['action_potential_electrode']
        physical_time = simulation_results['physical_time']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            electrogram_unipolar = simulation_results['electrogram_unipolar']
        else:
            electrogram_unipolar = None

        # show some action potentials and electrograms
        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(12, 8), sharex='col', sharey=False
        )

        for i, eid in enumerate([simulation_parameters['voxel_id_of_electrode'][0], simulation_parameters['voxel_id_of_electrode'][1], simulation_parameters['voxel_id_of_electrode'][2]]):
            # left column: action potentials
            axes[i, 0].plot(physical_time, action_potential[:, eid])
            axes[i, 0].set_title(f'Action Potential at Location {eid}')
            axes[i, 0].set_ylabel('Voltage (scaled)')
            axes[i, 0].set_xlabel('Time (ms)')

            # right column: unipolar electrograms
            if simulation_parameters['compute_electrogram_flag'] == 1:
                axes[i, 1].plot(physical_time, electrogram_unipolar[:, i])
                axes[i, 1].set_title(f'Unipolar Electrogram at Location {simulation_parameters['voxel_id_of_electrode'][i]}')
                axes[i, 1].set_ylabel('Voltage (scaled)')
                axes[i, 1].set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig(directory['result'] / f'{name_prefix}_ap_egm_{simulation_parameters['heart_model_flag']}_{simulation_parameters['arrhythmia_flag']}_{s1}_{s2[0]}.png', dpi=300)
        plt.close()

    # display simulation movie
    do_flag = 1
    if do_flag == 1:
        # load simulation results
        if str(s2) == '[]':
            simulation_results = dict(np.load(directory['result'] / f'{name_prefix}_simulation_results_{str(s1)}.npz', allow_pickle=False))
        else:
            simulation_results = dict(np.load(directory['result'] / f'{name_prefix}_simulation_results_{str(s1)}_{str(s2[0])}.npz', allow_pickle=False))

        save_movie_flag = 1 # 1: save movie. 0: do not save movie
        starting_time = 0 # 0 # ms
        ending_time = [] # ms. []: till the end. or specify a value

        simulation_results_file_name = directory['result'] / f'{name_prefix}_simulation_results_{str(s1)}_{str(s2[0])}.gif'
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
