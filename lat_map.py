import numpy as np # pip install numpy
import modules
import toolbox

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

def execute(node, result_folder, focal_1, focal_2, plot_lat_map_flag):
    # load simulation results
    if str(focal_2) == '[]':
        simulation_results = dict(np.load(result_folder / f'simulation_results_{focal_1}.npz', allow_pickle=False))
    else: 
        simulation_results = dict(np.load(result_folder / f'simulation_results_{focal_1}_{focal_2}.npz', allow_pickle=False))
    
    electrogram_unipolar = simulation_results['electrogram_unipolar']

    # compute and plot local activation time map
    if str(focal_2) == '[]':
        fig_name = result_folder / 'lat map' / f'lat_{str(focal_1)}.png'
    else:
        fig_name = result_folder / 'lat map' / f'lat_{str(focal_1)}_{str(focal_2)}.png'

    data_flag = 1 # 0: action potential, 1: electrogram
    geometry_flag = 2 # 2: 3D atrium
    lat = toolbox.lat_map_on_node.execute(node, electrogram_unipolar, data_flag, geometry_flag, plot_lat_map_flag, fig_name)

    # save local activation time
    if str(focal_2) == '[]':
        np.savez(result_folder / f'lat_{focal_1}.npz', lat)
    else:
        np.savez(result_folder / f'lat_{focal_1}_{focal_2}.npz', lat)

    common.crop_image.execute(fig_name)
