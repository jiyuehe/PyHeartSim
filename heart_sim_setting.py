import numpy as np # pip install numpy

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def execute(directory, geometry_name, result_folder):
    # load geometry data
    file_path = directory['data'] / geometry_name
    data = np.load(file_path, allow_pickle=True)
    geometry_data = {k: data[k] for k in data.files}

    n_node = geometry_data['voxel'].shape[0]

    e_id = np.arange(n_node) # use all nodes as electrode locations

    debug_plot = 0
    if debug_plot == 1: # show geometry voxel
        voxel = geometry_data['voxel']
        # Create color array: red for all nodes, blue for electrode nodes
        colors = np.array(['black'] * n_node)
        colors[e_id] = 'blue'
        # Create size array: size 1 for all nodes, size 10 for electrode nodes
        sizes = np.ones(n_node) * 3
        sizes[e_id] = 5
        fig = go.Figure(data=[go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=sizes, color=colors),
                showlegend=False)])
        fig.update_layout(scene=dict(aspectmode='data')) # set aspect ratio to be equal
        fig.show()

    # simulation parameters
    simulation_parameters = {
        'geometry_flag': 1, 
        # 0: 2D
        # 1: 3D
        # 2: long slab for computing conduction velocity
        'heart_model_flag': 0, 
        # 0: Mitchell-Schaeffer model
        # 1: Aliev-Panfilov model
        'arrhythmia_flag': 6, 
        # 0: focal
        # 1: rotor
        # 2: fibrillation
        # 3: for debugging, manually assign s1 and s2 region 
        # 4: 2 focal 2 locations both at 0ms
        # 5: 2 focal 2 locations 300ms apart
        # 6: 2 focal 2 locations 50ms apart
        'compute_electrogram_flag': 1, 
        # 1: compute electrogram 
        # 0: do not compute electrogram
        'electrode_id': e_id, # electrode locations for computing electrograms
        't_final': 1000, # ms
        'dt': 0.5, # ms. if dt is too large, simulation will become numerically unstable
    }

    save_result_flag = 1 # 1: save simulation results, 0: do not save simulation results

    input_arguments = {}
    input_arguments['geometry_data'] = geometry_data
    input_arguments['simulation_parameters'] = simulation_parameters
    input_arguments['save_result_flag'] = save_result_flag
    input_arguments['result_folder'] = result_folder

    return input_arguments
