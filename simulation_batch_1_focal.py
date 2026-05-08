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
import numpy as np # pip install numpy
import os
import simulation_individual
import utility
import common
import configuration

#%% 
mesh_id = 6 # finished 0-3, doing 4-5
name_prefix = configuration.mesh_name(mesh_id) # get mesh name prefix
directory = configuration.directory_setup() # set up directories

n_simulations = 1000
save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results
plot_lat_map_flag = 1 # 1: plot local activation time map. 0: do not plot local activation time map

# load geometry data
file_path = directory['data'] / f'{name_prefix}_clinical_data.npz'
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}

simulation_parameters, arrhythmia_parameters, heart_model_parameters = configuration.assign_simulation_parameters(name_prefix, geometry_data, [], [])

simulation_parameters['save_action_potential_of_all_voxel_flag'] = 0

input_arguments = {}
input_arguments['name_prefix'] = name_prefix
input_arguments['geometry_data'] = geometry_data
input_arguments['save_result_flag'] = save_result_flag
input_arguments['result_folder'] = directory['result']
input_arguments['result_folder'].mkdir(exist_ok=True) # create the folder if it does not exist
(input_arguments['result_folder'] / 'activation_maps').mkdir(exist_ok=True) # create the folder if it does not exist
input_arguments['simulation_parameters'] = simulation_parameters
input_arguments['arrhythmia_parameters'] = arrhythmia_parameters
input_arguments['heart_model_parameters'] = heart_model_parameters

#%%
# run simulations
voxel_electrode = geometry_data['voxel3mm_1mm_spacing']

s1_electrode = np.random.choice(np.arange(0,voxel_electrode.shape[0]), size=n_simulations, replace=False) # random ids in voxel_id_of_simulation_electrode without replacement
s1 = geometry_data['voxel_id_of_simulation_electrode'][s1_electrode] # convert to voxel ids

for loop_id in range(n_simulations): # 0 to n_simulations-1
    print(f'===== simulation set {loop_id+1} of {n_simulations} =====')

    file_name = f'{name_prefix}_simulation_results_{s1[loop_id]}.npz'

    if not os.path.exists(input_arguments['result_folder'] / file_name):
        input_arguments['s1'] = s1[loop_id]
        input_arguments['s2'] = []
        input_arguments['arrhythmia_parameters']['s1_pacing_voxel_id'] = s1[loop_id]
        simulation_individual.run_simulation(input_arguments)

        # display simulation movie
        do_flag = 0
        if do_flag == 1:
            # load simulation results
            simulation_results = dict(np.load(input_arguments['result_folder'] / file_name, allow_pickle=False))

            save_movie_flag = 1 # 1: save movie. 0: do not save movie
            starting_time = 0 # 0 # ms
            ending_time = 1000 # ms. []: till the end. or specify a value

            simulation_results_file_name = input_arguments['result_folder'] / f'{name_prefix}_simulation_results_{str(s1[loop_id])}.gif'
            movie_save_dir = input_arguments['result_folder'] / simulation_results_file_name

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

        # compute local activation time
        simulation_results = dict(np.load(input_arguments['result_folder'] / file_name, allow_pickle=False)) # load simulation results
        electrogram_unipolar = simulation_results['electrogram_unipolar']
        
        lat_electrode = utility.lat_map.compute_electrode_lat(electrogram_unipolar)

        # plot local activation time map
        if plot_lat_map_flag == 1:
            fig_name = input_arguments['result_folder'] / 'activation_maps' / f'{name_prefix}_lat_{str(s1[loop_id])}.png'
            
            geometry_flag = simulation_results['geometry_flag']
            utility.lat_map.plot(voxel_electrode, lat_electrode, geometry_flag, fig_name)
            common.crop_image(fig_name)

        # save lat to simulation_results
        simulation_results['lat_electrode'] = lat_electrode
        np.savez(input_arguments['result_folder'] / file_name, **simulation_results)

#%%
print('done')

# ensures the kernel dies. 
# because even after the visual studio code is closed, 
# there will still be heavy python process in CPU cause computer to heat up and slow down.
os._exit(0) 

#%%