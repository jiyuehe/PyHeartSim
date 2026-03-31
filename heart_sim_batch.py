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
import toolbox.codes
import heart_sim_individual
import numpy as np # pip install numpy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt # pip install matplotlib

import lat_map

#%% 
directory = {}
directory['home'] = script_dir
directory['data'] = script_dir.parent / '0_data'
directory['result'] = Path('/home/j/Desktop/hdd/103_1-lagood_3mm')

geometry_name = '103_1-lagood_geometry.npz'

# load geometry data
file_path = directory['data'] / geometry_name
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}

simulation_parameters = modules.setting.assign_simulation_parameters(geometry_data)

input_arguments = {}
input_arguments['geometry_data'] = geometry_data
input_arguments['save_result_flag'] = 1
input_arguments['result_folder'] = directory['result']
input_arguments['simulation_parameters'] = simulation_parameters

# save s1 and s2 to text file
n_simulations = 1
n_nodes = input_arguments['geometry_data']['voxel'].shape[0]
s1 = np.random.choice(n_nodes, size=n_simulations) # random integers from 0 to n_nodes-1
s2 = s1 # np.random.choice(n_nodes, size=n_simulations) # random integers from 0 to n_nodes-1

#%%
plot_lat_map_flag = 1

geometry = {}
geometry['vertex'] = geometry_data['vertex3mm']
geometry['face'] = geometry_data['face3mm']
voxel_id_of_vertex3mm = geometry_data['voxel_id_of_vertex3mm']
geometry['node'] = geometry_data['voxel'][voxel_id_of_vertex3mm, :]

for loop_id in range(n_simulations): # 0 to n_simulations-1
    print(f'===== simulation set {loop_id+1} of {n_simulations} =====')

    # focal 1
    focal_1 = s1[loop_id]
    focal_2 = []
    if not os.path.exists(directory['result'] / f'lat_{str(focal_1)}.npz'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.run_simulation(input_arguments)
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # focal 2
    focal_1 = s2[loop_id]
    focal_2 = []
    if not os.path.exists(directory['result'] / f'lat_{str(focal_1)}.npz'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.run_simulation(input_arguments)
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # dual focals
    focal_1 = s1[loop_id]
    focal_2 = s2[loop_id]
    if not os.path.exists(directory['result'] / f'simulation_results_{str(focal_1)}_{str(focal_2)}.npz'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.run_simulation(input_arguments)
    lat_map.execute(geometry, directory['result'], focal_1, focal_2, plot_lat_map_flag)

#%%
# ensures the kernel dies. 
# because even after the visual studio code is closed, 
# there will still be heavy python process in CPU cause computer to heat up and slow down.
os._exit(0) 

#%%