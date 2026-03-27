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

import modules
import toolbox
import numpy as np # pip install numpy
from numba import cuda # pip install numba
import time
import matplotlib.pyplot as plt # pip install matplotlib

import lat_map

#%%
def run_simulation(input_arguments):
    simulation_parameters = input_arguments['simulation_parameters']
    s1 = input_arguments['s1']
    s2 = input_arguments['s2']
    save_result_flag = input_arguments['save_result_flag']
    result_folder = input_arguments['result_folder']
    geometry_data = input_arguments['geometry_data']

    n_voxel = geometry_data['voxel'].shape[0]

    # arrhythmia parameters
    arrhythmia_parameters = modules.setting.arrhythmia_parameters(simulation_parameters, s1, s2, script_dir)

    # heart model parameters
    heart_model_parameter = modules.setting.heart_model_parameters(simulation_parameters, n_voxel)

    # scale the time
    simulation_parameters, arrhythmia_parameters, heart_model_parameter = modules.setting.scale_heart_model_time(simulation_parameters, arrhythmia_parameters, heart_model_parameter)

    if simulation_parameters['geometry_flag'] == 2: # long slab for computing conduction velocity
        x_coordinates = geometry_data['voxel'][:, 0]
        voxels_1 = np.where(x_coordinates == np.min(x_coordinates))[0]
        # voxels_2 = np.where(x_coordinates == np.max(x_coordinates))[0]
        arrhythmia_parameters['s1_pacing_voxel_id'] = voxels_1

    # fiber orientations. will be used in computing heart simulation and unipolar electrogram
    D0 = modules.load_fiber.execute(n_voxel)

    # compute heart model equation parts
    P_2d = modules.heart_model_equation_parts.execute(simulation_parameters, n_voxel, D0, geometry_data['neighbor_id_2d'], heart_model_parameter)

    # solve differential equations
    start = time.time()
    if cuda.is_available(): # GPU parallel
        print("GPU parallel computing for simulation")
        action_potential, h, physical_time = modules.compute_simulation_gpu.execute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters)
    end = time.time()
    print(f'simulation completed in {end - start:.1f} seconds')

    # compute unipolar electrogram
    start = time.time()
    if simulation_parameters['compute_electrogram_flag'] == 1:
        voxel = geometry_data['voxel']
        electrode_id = simulation_parameters['electrode_id']
        electrode_xyz = voxel[electrode_id, :]
        Delta = geometry_data['Delta']
        neighbor_id_2d = geometry_data['neighbor_id_2d']
        
        # Check GPU availability and choose appropriate module
        if cuda.is_available():
            print("GPU parallel computing for electrogram")
            electrogram_unipolar = modules.compute_unipolar_electrogram_gpu.execute(electrode_xyz, voxel, D0, heart_model_parameter['c_voxel'], action_potential, Delta, neighbor_id_2d)
    end = time.time()
    print(f'electrogram computation completed in {end - start:.1f} seconds')

    # save simulation results
    if save_result_flag == 1:
        voxel_for_each_vertex_3mm = geometry_data['voxel_for_each_vertex_3mm']

        simulation_results = {}
        simulation_results['action_potential'] = action_potential[:, voxel_for_each_vertex_3mm] # shape: (time, n_vertex_3mm)
        simulation_results['h'] = h[:, voxel_for_each_vertex_3mm] # shape: (time, n_vertex_3mm)
        simulation_results['physical_time'] = physical_time
        simulation_results['geometry_flag'] = simulation_parameters['geometry_flag']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            simulation_results['electrode_id'] = simulation_parameters['electrode_id']
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
    directory['data'] = script_dir.parent / '0_data'
    directory['result'] = directory['home'] / 'result'
    
    geometry_name = '103_1-lagood_geometry.npz'

    # load geometry data
    file_path = directory['data'] / geometry_name
    data = np.load(file_path, allow_pickle=False)
    geometry_data = {k: data[k] for k in data.files}

    save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results

    input_arguments = {}
    input_arguments['geometry_data'] = geometry_data
    input_arguments['save_result_flag'] = save_result_flag
    input_arguments['result_folder'] = directory['result']

    s1 = 1000
    s2 = 1000
    input_arguments['s1'] = s1
    input_arguments['s2'] = s2

    # run simulation
    run_simulation(input_arguments)

    focal_1 = s1
    focal_2 = s2
    plot_lat_map_flag = 1
    
    geometry = {}
    geometry['vertex'] = input_arguments['geometry_data']['vertex_3mm']
    geometry['face'] = input_arguments['geometry_data']['face_3mm']
    voxel_for_each_vertex_3mm = input_arguments['geometry_data']['voxel_for_each_vertex_3mm']
    geometry['node'] = input_arguments['geometry_data']['voxel'][voxel_for_each_vertex_3mm, :]
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # plot some action potentials and electrograms
    do_flag = 1
    if do_flag == 1: 
        simulation_parameters = input_arguments['simulation_parameters']

        # load simulation results
        if str(s2) == '[]':
            sim_data = dict(np.load(directory['result'] / f'simulation_results_{str(s1)}.npz', allow_pickle=False))
        else:
            sim_data = dict(np.load(directory['result'] / f'simulation_results_{str(s1)}_{str(s2)}.npz', allow_pickle=False))

        action_potential = sim_data['action_potential']
        physical_time = sim_data['physical_time']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            electrogram_unipolar = sim_data['electrogram_unipolar']
        else:
            electrogram_unipolar = None

        # show some action potentials and electrograms
        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(12, 8), sharex='col', sharey=False
        )

        for i, eid in enumerate([simulation_parameters['electrode_id'][0], simulation_parameters['electrode_id'][1], simulation_parameters['electrode_id'][2]]):
            # left column: action potentials
            axes[i, 0].plot(physical_time, action_potential[:, eid])
            axes[i, 0].set_title(f'Action Potential at Location {eid}')
            axes[i, 0].set_ylabel('Voltage (scaled)')
            axes[i, 0].set_xlabel('Time (ms)')

            # right column: unipolar electrograms
            if simulation_parameters['compute_electrogram_flag'] == 1:
                axes[i, 1].plot(physical_time, electrogram_unipolar[:, i])
                axes[i, 1].set_title(f'Unipolar Electrogram at Location {simulation_parameters['electrode_id'][i]}')
                axes[i, 1].set_ylabel('Voltage (scaled)')
                axes[i, 1].set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig(directory['result'] / f'ap_egm_{simulation_parameters["heart_model_flag"]}_{simulation_parameters["arrhythmia_flag"]}_{s1}_{s2}.png', dpi=300)
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
        in_arg['geometry_data'] = input_arguments['geometry_data']
        toolbox.display_simulation_movie.execute(in_arg)

#%%
