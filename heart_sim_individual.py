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

# add the workspace root to Python path
import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import common

import simulation
import toolbox
import numpy as np # pip install numpy
from numba import cuda # pip install numba
import time
import matplotlib.pyplot as plt # pip install matplotlib

import lat_map

#%%
def run_simulation(input_arguments):
    s1 = input_arguments['s1']
    s2 = input_arguments['s2']
    save_result_flag = input_arguments['save_result_flag']
    result_folder = input_arguments['result_folder']
    geometry_data = input_arguments['geometry_data']

    n_voxel = geometry_data['voxel'].shape[0]

    # parameters
    simulation_parameters = input_arguments['simulation_parameters']
    arrhythmia_parameters = simulation.setting.assign_arrhythmia_parameters(simulation_parameters, s1, s2, script_dir)
    heart_model_parameters = simulation.setting.assign_heart_model_parameters(simulation_parameters, n_voxel)
    simulation_parameters, arrhythmia_parameters, heart_model_parameters = simulation.setting.scale_heart_model_time(simulation_parameters, arrhythmia_parameters, heart_model_parameters)

    if simulation_parameters['geometry_flag'] == 2: # long slab for computing conduction velocity
        x_coordinates = geometry_data['voxel'][:, 0]
        voxels_1 = np.where(x_coordinates == np.min(x_coordinates))[0]
        # voxels_2 = np.where(x_coordinates == np.max(x_coordinates))[0]
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
        voxel_id_of_voxel3mm = geometry_data['voxel_id_of_voxel3mm']

        simulation_results = {}
        simulation_results['action_potential_voxel3mm'] = action_potential[:, voxel_id_of_voxel3mm] # shape: (time, n_voxel3mm)
        simulation_results['h_voxel3mm'] = h[:, voxel_id_of_voxel3mm] # shape: (time, n_voxel3mm)
        simulation_results['physical_time'] = physical_time
        simulation_results['geometry_flag'] = simulation_parameters['geometry_flag']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            simulation_results['voxel_id_of_electrode'] = simulation_parameters['voxel_id_of_electrode']
            simulation_results['electrogram_unipolar'] = electrogram_unipolar

        # save simulation results
        if str(s1) != '[]' and str(s2) == '[]':
            np.savez(result_folder / f'simulation_results_{str(s1)}', **simulation_results)
        elif str(s1) == '[]' and str(s2) == '[]':
            np.savez(result_folder / 'simulation_results', **simulation_results)
        elif str(s1) != '[]' and str(s2) != '[]':
            np.savez(result_folder / f'simulation_results_{str(s1)}_{str(s2)}', **simulation_results)

#%%
# NOTE: 
# If running this script directly, the following code block will be executed. 
# If calling the execute() function from another script, the following code block will be ignored.
if __name__ == "__main__":
    directory = {}
    directory['home'] = script_dir
    directory['result'] = script_dir.parent / '0_result'
    directory['data'] = script_dir.parent / '0_data'
    geometry_name = '103_1-lagood_geometry.npz'
    save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results
    plot_lat_map_flag = 1

    # create the folder if it does not exist
    directory['result'].mkdir(exist_ok=True)

    # load geometry data
    file_path = directory['data'] / geometry_name
    data = np.load(file_path, allow_pickle=False)
    geometry_data = {k: data[k] for k in data.files}

    input_arguments = {}
    input_arguments['geometry_data'] = geometry_data
    input_arguments['save_result_flag'] = save_result_flag
    input_arguments['result_folder'] = directory['result']

    s1 = 1000
    s2 = 1000
    input_arguments['s1'] = s1
    input_arguments['s2'] = s2

    simulation_parameters = simulation.setting.assign_simulation_parameters(geometry_data)
    input_arguments['simulation_parameters'] = simulation_parameters

    # run simulation
    run_simulation(input_arguments)

    focal_1 = s1
    focal_2 = s2
    
    geometry = {}
    # geometry['vertex'] = geometry_data['vertex3mm']
    # geometry['face'] = geometry_data['face3mm']
    voxel_id_of_voxel3mm = geometry_data['voxel_id_of_voxel3mm']
    geometry['node'] = geometry_data['voxel'][voxel_id_of_voxel3mm, :]
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # plot some action potentials and electrograms
    do_flag = 1
    if do_flag == 1: 
        # load simulation results
        if str(s2) == '[]':
            sim_data = dict(np.load(directory['result'] / f'simulation_results_{str(s1)}.npz', allow_pickle=False))
        else:
            sim_data = dict(np.load(directory['result'] / f'simulation_results_{str(s1)}_{str(s2)}.npz', allow_pickle=False))

        action_potential = sim_data['action_potential_voxel3mm']
        physical_time = sim_data['physical_time']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            electrogram_unipolar = sim_data['electrogram_unipolar']
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
        plt.savefig(directory['result'] / f'ap_egm_{simulation_parameters['heart_model_flag']}_{simulation_parameters['arrhythmia_flag']}_{s1}_{s2}.png', dpi=300)
        plt.close()

    # display simulation movie
    do_flag = 1
    if do_flag == 1:
        save_movie_flag = 1 # 1: save movie. 0: do not save movie
        starting_time = 0 # 0 # ms
        ending_time = [] # ms. []: till the end. or specify a value
        sim_file_name = f'simulation_results_{str(s1)}_{str(s2)}.npz'
        simulation_results_file_name = directory['result'] / sim_file_name
        movie_save_dir = directory['result'] / sim_file_name.replace('.npz', '.gif')
        simulation_results = dict(np.load(directory['result'] / simulation_results_file_name, allow_pickle=False)) # load simulation results
        in_arg = {}
        in_arg['save_movie_flag'] = save_movie_flag
        in_arg['starting_time'] = starting_time
        in_arg['ending_time'] = ending_time
        in_arg['simulation_results_file_name'] = simulation_results_file_name
        in_arg['movie_save_dir'] = movie_save_dir
        in_arg['simulation_results'] = simulation_results
        in_arg['geometry_data'] = geometry_data
        toolbox.display_simulation_movie.execute(in_arg)

#%%
