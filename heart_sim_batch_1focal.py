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
import toolbox.codes
import heart_sim_individual
import numpy as np # pip install numpy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # pip install matplotlib

import utility.lat_map as lat_map

#%% 
directory = {}
directory['home'] = script_dir
directory['result'] = Path('/home/j/Desktop/hdd/103_1-lagood_3mm_1focal')
directory['data'] = script_dir.parent / '0_data'

# create the folder if it does not exist
directory['result'].mkdir(exist_ok=True)

geometry_name = '103_1-lagood_geometry.npz'

# load geometry data
file_path = directory['data'] / geometry_name
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}

simulation_parameters = simulation.setting.assign_simulation_parameters(geometry_data)

input_arguments = {}
input_arguments['geometry_data'] = geometry_data
input_arguments['save_result_flag'] = 1
input_arguments['result_folder'] = directory['result']
input_arguments['simulation_parameters'] = simulation_parameters

# save s1 and s2 to text file
n_simulations = 3000
voxel_id_of_voxel3mm = geometry_data['voxel_id_of_voxel3mm']

s1 = np.random.choice(voxel_id_of_voxel3mm, size=n_simulations, replace=False) # random ids in voxel_id_of_voxel3mm without replacement

#%%
plot_lat_map_flag = 1

geometry = {}
geometry['node'] = geometry_data['voxel'][voxel_id_of_voxel3mm, :]

for loop_id in range(n_simulations): # 0 to n_simulations-1
    print(f'===== simulation set {loop_id+1} of {n_simulations} =====')

    focal_1 = s1[loop_id]
    focal_2 = []
    if not os.path.exists(directory['result'] / f'lat_{str(focal_1)}.npz'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.run_simulation(input_arguments)
        
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

#%%
print('done')

# ensures the kernel dies. 
# because even after the visual studio code is closed, 
# there will still be heavy python process in CPU cause computer to heat up and slow down.
os._exit(0) 

#%%