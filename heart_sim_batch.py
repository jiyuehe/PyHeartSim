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

import heart_sim_setting
import lat_map

#%% 
directory = {}
directory['home'] = script_dir
directory['data'] = script_dir.parent / '0_data'
directory['result'] = Path('/home/j/Desktop/hdd/atrium 2, 2 focal 2 location 15ms apart')

geometry_name = '6-1-1-1-DP W CS 3 4 270CL_CS 13 14 300_processed.npy'
input_arguments = heart_sim_setting.execute(directory, geometry_name, directory['result'])

# save s1 and s2 to text file
n_simulations = 2000
n_nodes = input_arguments['geometry_data']['voxel'].shape[0]
s1 = np.random.choice(n_nodes, size=n_simulations) # random integers from 0 to n_nodes-1
s2 = np.random.choice(n_nodes, size=n_simulations) # random integers from 0 to n_nodes-1

#%%
plot_lat_map_flag = 1
node = input_arguments['geometry_data']['voxel']
for loop_id in range(n_simulations): # 0 to n_simulations-1
    print(f'===== simulation set {loop_id+1} of {n_simulations} =====')

    # focal 1
    focal_1 = s1[loop_id]
    focal_2 = []
    if not os.path.exists(directory['result'] / f'lat_{str(focal_1)}.npy'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.execute(input_arguments)
    lat_map.execute(node, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # focal 2
    focal_1 = s2[loop_id]
    focal_2 = []
    if not os.path.exists(directory['result'] / f'lat_{str(focal_1)}.npy'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.execute(input_arguments)
    lat_map.execute(node, directory['result'], focal_1, focal_2, plot_lat_map_flag)

    # dual focals
    focal_1 = s1[loop_id]
    focal_2 = s2[loop_id]
    if not os.path.exists(directory['result'] / f'simulation_results_{str(focal_1)}_{str(focal_2)}.npy'):
        input_arguments['s1'] = focal_1
        input_arguments['s2'] = focal_2
        heart_sim_individual.execute(input_arguments)
    lat_map.execute(node, directory['result'], focal_1, focal_2, plot_lat_map_flag)

#%%
# ensures the kernel dies. 
# because even after the visual studio code is closed, 
# there will still be heavy python process in CPU cause computer to heat up and slow down.
os._exit(0) 

#%%