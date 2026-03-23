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

import heart_sim_setting
import lat_map

#%%
def execute(input_arguments):
    simulation_parameters = input_arguments['simulation_parameters']
    s1 = input_arguments['s1']
    s2 = input_arguments['s2']
    save_result_flag = input_arguments['save_result_flag']
    result_folder = input_arguments['result_folder']
    geometry_data = input_arguments['geometry_data']

    n_voxel = geometry_data['voxel'].shape[0]

    # arrhythmia parameters
    arrhythmia_parameters = modules.load_parameters.arrhythmia_parameters(simulation_parameters, s1, s2, script_dir)

    # heart model parameters
    heart_model_parameter = modules.load_parameters.heart_model_parameters(simulation_parameters, n_voxel)

    # scale the time
    simulation_parameters, arrhythmia_parameters, heart_model_parameter = modules.scale_time.execute(simulation_parameters, arrhythmia_parameters, heart_model_parameter)

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
        simulation_results = {}
        simulation_results['action_potential'] = action_potential
        simulation_results['h'] = h
        simulation_results['physical_time'] = physical_time
        simulation_results['geometry_flag'] = simulation_parameters['geometry_flag']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            simulation_results['electrode_id'] = simulation_parameters['electrode_id']
            simulation_results['electrogram_unipolar'] = electrogram_unipolar

        # save simulation results
        if str(s1) != '[]' and str(s2) == '[]':
            np.save(result_folder / f'simulation_results_{str(s1)}', simulation_results)
        elif str(s1) == '[]' and str(s2) == '[]':
            np.save(result_folder / 'simulation_results', simulation_results)
        elif str(s1) != '[]' and str(s2) != '[]':
            np.save(result_folder / f'simulation_results_{str(s1)}_{str(s2)}', simulation_results)

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
    input_arguments = heart_sim_setting.execute(directory, geometry_name, directory['result'])

    # # ==============================
    # # find the pacing locations
    # activation_uni = input_arguments['geometry_data']['activation_uni']
    # activation_uni = activation_uni.astype(float)
    # activation_uni[activation_uni==0] = np.nan
    # electrode_node_id = input_arguments['geometry_data']['electrode_node_id']
    # node = input_arguments['geometry_data']['voxel']
    # electrode_nodes = node[electrode_node_id, :]

    # # find pacing electrodes
    # pacing_electrodes_id = np.where(activation_uni <= np.nanmin(activation_uni)+5)
    # pacing_node_id = electrode_node_id[pacing_electrodes_id]
    # pacing_nodes = node[pacing_node_id, :]

    # # cluster the pacing_nodes into 2 clusters based on distance
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # cluster_labels = kmeans.fit_predict(pacing_nodes)
    # cluster_centers = kmeans.cluster_centers_

    # # find the node id that is nearest to each cluster center
    # from scipy.spatial.distance import cdist
    # distances = cdist(cluster_centers, node)
    # nearest_node_ids = np.argmin(distances, axis=1)

    # debug_plot = 0
    # if debug_plot == 1:
    #     data = activation_uni
    #     data_min = np.nanmin(data)
    #     data_max = np.nanmax(data)
    #     data_threshold = data_min - 0.01
    #     map_color = common.convert_data_to_color.execute(data, data_min, data_max, data_threshold)

    #     # use plotly to display the electrode nodes and assign them map_color
    #     import plotly.graph_objects as go
    #     fig = go.Figure(data=[go.Scatter3d(
    #         x=electrode_nodes[:, 0],
    #         y=electrode_nodes[:, 1],
    #         z=electrode_nodes[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=5,
    #             color=map_color
    #         ),
    #         name='Electrodes'
    #     )])
        
    #     # Add red cross markers for pacing electrodes
    #     fig.add_trace(go.Scatter3d(
    #         x=pacing_nodes[:, 0],
    #         y=pacing_nodes[:, 1],
    #         z=pacing_nodes[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=4,
    #             color='red',
    #             symbol='x'
    #         ),
    #         name='Pacing Electrodes'
    #     ))
        
    #     # Plot cluster centers with black cross
    #     fig.add_trace(go.Scatter3d(
    #         x=node[nearest_node_ids, 0],
    #         y=node[nearest_node_ids, 1],
    #         z=node[nearest_node_ids, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=4,
    #             color='black',
    #             symbol='x'
    #         ),
    #         name='Cluster Centers'
    #     ))
        
    #     fig.update_layout(
    #         title='Electrode Nodes with Activation Color',
    #         scene=dict(
    #             xaxis_title='X',
    #             yaxis_title='Y',
    #             zaxis_title='Z'
    #         )
    #     )
    #     fig.show()
    # # ==============================

    # s1 = nearest_node_ids[0] # node id for s1 pacing
    # s2 = nearest_node_ids[1] # node id for s2 pacing
    node = input_arguments['geometry_data']['voxel']
    s1 = 1000
    s2 = 1000
    input_arguments['s1'] = s1
    input_arguments['s2'] = s2

    # run simulation
    execute(input_arguments)

    focal_1 = s1
    focal_2 = s2
    plot_lat_map_flag = 1
    lat_map.execute(node, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # plot some action potentials and electrograms
    do_flag = 1
    if do_flag == 1: 
        simulation_parameters = input_arguments['simulation_parameters']

        # load simulation results
        if str(s2) == '[]':
            sim_data = np.load(directory['result'] / f'simulation_results_{str(s1)}.npy', allow_pickle=True)
        else:
            sim_data = np.load(directory['result'] / f'simulation_results_{str(s1)}_{str(s2)}.npy', allow_pickle=True)
        sim_data = sim_data.item()

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
        sim_file_name = f'simulation_results_{str(s1)}_{str(s2)}.npy'
        simulation_results_file_name = directory['result'] / sim_file_name
        movie_save_dir = directory['result'] / sim_file_name.replace('.npy', '.gif') 
        simulation_results = np.load(directory['result'] / simulation_results_file_name, allow_pickle=True) # load simulation results
        simulation_results = simulation_results.item()
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
