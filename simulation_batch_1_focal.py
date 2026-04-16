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
import configuration

#%% 
directory = configuration.directory_setup() # set up directories
name_prefix = configuration.mesh_name() # get mesh name prefix

n_simulations = 20
save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results
plot_lat_map_flag = 1 # 1: plot local activation time map. 0: do not plot local activation time map

# load geometry data
file_path = directory['data'] / f'{name_prefix}_geometry.npz'
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}

voxel = geometry_data['voxel']
n_voxel = voxel.shape[0]
electrode_voxel = voxel[geometry_data['voxel_id_of_electrode'], :]

voxel_id_of_electrode = geometry_data['voxel_id_of_electrode']
s1 = np.random.choice(voxel_id_of_electrode, size=n_simulations, replace=False) # random ids in voxel_id_of_electrode without replacement
s2 = []

simulation_parameters, arrhythmia_parameters, heart_model_parameters = configuration.assign_simulation_parameters(name_prefix, geometry_data, s1, s2, n_voxel)

input_arguments = {}
input_arguments['name_prefix'] = name_prefix
input_arguments['geometry_data'] = geometry_data
input_arguments['save_result_flag'] = save_result_flag
input_arguments['result_folder'] = directory['result']
input_arguments['simulation_parameters'] = simulation_parameters
input_arguments['arrhythmia_parameters'] = arrhythmia_parameters
input_arguments['heart_model_parameters'] = heart_model_parameters

#%%
# run simulations
for loop_id in range(n_simulations): # 0 to n_simulations-1
    print(f'===== simulation set {loop_id+1} of {n_simulations} =====')

    if not os.path.exists(directory['result'] / f'lat_{str(s1[loop_id])}.npz'):
        input_arguments['s1'] = s1[loop_id]
        input_arguments['s2'] = s2
        simulation_individual.run_simulation(input_arguments)

        # compute local activation time
        file_name = f'{name_prefix}_simulation_results_{s1[loop_id]}.npz'
        simulation_results = dict(np.load(directory['result'] / file_name, allow_pickle=False)) # load simulation results
        electrogram_unipolar = simulation_results['electrogram_unipolar']
        lat_electrode = utility.lat_map.compute_electrode_lat(electrogram_unipolar)

        # interpolate local activation time from electrode locations to all voxels
        lat_voxel = utility.lat_map.interpolate_lat(voxel, electrode_voxel, lat_electrode)

        # plot local activation time map
        if plot_lat_map_flag == 1:
            fig_name = directory['result'] / f'{name_prefix}_lat_{str(s1[loop_id])}.png'
            
            geometry_flag = simulation_results['geometry_flag']
            utility.lat_map.plot(voxel, lat_voxel, geometry_flag, fig_name)
            utility.common.crop_image(fig_name)

        # save lat to simulation_results
        simulation_results['lat_electrode'] = lat_electrode
        np.savez(directory['result'] / file_name, **simulation_results)

#%%
print('done')

# ensures the kernel dies. 
# because even after the visual studio code is closed, 
# there will still be heavy python process in CPU cause computer to heat up and slow down.
os._exit(0) 

#%%