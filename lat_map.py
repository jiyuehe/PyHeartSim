import numpy as np # pip install numpy
import common
import modules
import toolbox

def execute(node, result_folder, focal_1, focal_2, plot_lat_map_flag):
    # load simulation results
    if str(focal_2) == '[]':
        simulation_results = np.load(result_folder / f'simulation_results_{focal_1}.npy', allow_pickle=True)
    else: 
        simulation_results = np.load(result_folder / f'simulation_results_{focal_1}_{focal_2}.npy', allow_pickle=True)
    simulation_results = simulation_results.item()
    
    electrogram_unipolar = simulation_results['electrogram_unipolar']

    # compute and plot local activation time map
    if str(focal_2) == '[]':
        fig_name = result_folder / f'lat_{str(focal_1)}.png'
    else:
        fig_name = result_folder / f'lat_{str(focal_1)}_{str(focal_2)}.png'

    data_flag = 1 # 0: action potential, 1: electrogram
    geometry_flag = 2 # 2: 3D atrium
    lat = toolbox.lat_map_on_node.execute(node, electrogram_unipolar, data_flag, geometry_flag, plot_lat_map_flag, fig_name)

    # save local activation time
    if str(focal_2) == '[]':
        np.save(result_folder / f'lat_{focal_1}.npy', lat)
    else:
        np.save(result_folder / f'lat_{focal_1}_{focal_2}.npy', lat)

    common.corp_image.execute(fig_name)
