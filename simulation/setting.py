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

import numpy as np

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def assign_simulation_parameters(geometry_data):
    voxel_id_of_electrode = geometry_data['voxel_id_of_voxel3mm'] # assign electrode locations

    debug_plot = 0
    if debug_plot == 1: # show geometry voxel
        voxel = geometry_data['voxel']
        n_node = voxel.shape[0]

        colors = np.array(['black'] * n_node)
        colors[voxel_id_of_electrode] = 'blue'
        
        sizes = np.ones(n_node) * 3
        sizes[voxel_id_of_electrode] = 5
        fig = go.Figure(data=[go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=sizes, color=colors),
                showlegend=False)])
        fig.update_layout(scene=dict(aspectmode='data')) # set aspect ratio to be equal
        fig.show()

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
        'voxel_id_of_electrode': voxel_id_of_electrode, # electrode for computing electrograms
        't_final': 1000, # ms
        'dt': 0.5, # ms. 0.5 is good. if dt is too large, simulation will become numerically unstable
    }

    return simulation_parameters

def assign_arrhythmia_parameters(simulation_parameters, s1, s2, script_dir):
    # NOTE: changes of heart_model_parameters or pacing magnitude/duration will change the ap/h_min/max thresholds
    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        arrhythmia_parameters = {
            'pacing_start_time': 10, # ms
            'pacing_cycle_length': 300, # ms
            's1_pacing_voxel_id': s1, # node id for s1 pacing
            's2_pacing_voxel_id': s2, # node id for s2 pacing
            's1_t': 0, # won't be used, but keep it to be compatible with the code structure
            's1_s2_delta_t': 0, # won't be used, but keep it to be compatible with the code structure
            'ap_min': 0, # won't be used, but keep it to be compatible with the code structure
            'ap_max': 0, # won't be used, but keep it to be compatible with the code structure
            'h_min': 0, # won't be used, but keep it to be compatible with the code structure
            'h_max': 0, # won't be used, but keep it to be compatible with the code structure
            's2_region_size_factor': 0, # won't be used, but keep it to be compatible with the code structure
        }
    elif simulation_parameters['arrhythmia_flag'] == 1: # simple rotor
        arrhythmia_parameters = {
            'pacing_start_time': 0, # won't be used, but keep it to be compatible with the code structure
            'pacing_cycle_length': 0, # won't be used, but keep it to be compatible with the code structure
            's1_pacing_voxel_id': s1, # node id for s1 pacing
            's2_pacing_voxel_id': s2, # won't be used, but keep it to be compatible with the code structure
            's1_t': 0, # ms. time of s1 pacing
            's1_s2_delta_t': 230, # ms. time interval between s1 and s2
            'ap_min': 0.00038510505014280766, # a threshold value of action potential
            'ap_max': 0.07687293043769826, # a threshold value of action potential
            'h_min': 0.1623103413330824, # a threshold value of action potential
            'h_max': 0.39717038588499276,# a threshold value of action potential
            's2_region_size_factor': 0.7, # a less than 1 multiplication factor to reduce s2 pacing region size
        }
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        arrhythmia_parameters = {
            'pacing_start_time': 0, # won't be used, but keep it to be compatible with the code structure
            'pacing_cycle_length': 0, # won't be used, but keep it to be compatible with the code structure
            's1_pacing_voxel_id': s1, # node id for s1 pacing
            's2_pacing_voxel_id': s2, # won't be used, but keep it to be compatible with the code structure
            's1_t': 0, # ms. time of s1 pacing
            's1_s2_delta_t': 130, # ms. time interval between s1 and s2
            'ap_min': 0.05, # a threshold value of action potential
            'ap_max': 0.25, # a threshold value of action potential
            'h_min': 0.025, # a threshold value of action potential
            'h_max': 0.120,# a threshold value of action potential
            's2_region_size_factor': 0.5, # a less than 1 multiplication factor to reduce s2 pacing region size
        }
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging, manually assign s1 and s2 region
        arrhythmia_parameters = {
            'pacing_start_time': 0, # won't be used, but keep it to be compatible with the code structure
            'pacing_cycle_length': 0, # won't be used, but keep it to be compatible with the code structure
            's1_pacing_voxel_id': s1, # node id for s1 pacing
            's2_pacing_voxel_id': s2, # node id for s2 pacing
            's1_t': 0, # ms. time of s1 pacing
            's1_s2_delta_t': 230, # ms. time interval between s1 and s2
            'ap_min': 0, # won't be used, but keep it to be compatible with the code structure
            'ap_max': 0, # won't be used, but keep it to be compatible with the code structure
            'h_min': 0, # won't be used, but keep it to be compatible with the code structure
            'h_max': 0, # won't be used, but keep it to be compatible with the code structure
            's2_region_size_factor': 0, # won't be used, but keep it to be compatible with the code structure
        }

        flag = 1 # 0: debug fibrillation. 1: debug simple rotor
        if flag == 0:
            node_flag = np.load(script_dir / 'data' / 'node_flag_afib.npy')
        if flag == 1:
            node_flag = np.load(script_dir / 'data' / 'node_flag.npy')
        arrhythmia_parameters['s1_pacing_voxel_id'] = np.where(node_flag == 1)[0]

        focal_or_rotor_flag = 1 # 0: s1 only focal. 1: s1 s2 rotor
        if focal_or_rotor_flag == 0:
            arrhythmia_parameters['s2_pacing_voxel_id'] = []
        elif focal_or_rotor_flag == 1:
            arrhythmia_parameters['s2_pacing_voxel_id'] = np.where(node_flag == 2)[0]

    return arrhythmia_parameters

def assign_heart_model_parameters(simulation_parameters, n_voxel):
    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
            tau_in = 0.3
            tau_out = 6
            tau_open = 120
            tau_close = 80
            v_gate = 0.13
        elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
            k = 8.0
            a = 0.15
            epsilon_0 = 0.002
            mu1 = 0.2
            mu2 = 0.3
    elif simulation_parameters['arrhythmia_flag'] == 1: # simple rotor
        if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
            tau_in = 0.3
            tau_out = 6
            tau_open = 120
            tau_close = 80
            v_gate = 0.13
        elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
            k = 8.0
            a = 0.15
            epsilon_0 = 0.002
            mu1 = 0.2
            mu2 = 0.3
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
            tau_in = 0.08
            tau_out = 6
            tau_open = 80
            tau_close = 30
            v_gate = 0.13
        elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
            k = 8.0
            a = 0.15
            epsilon_0 = 0.002
            mu1 = 0.2
            mu2 = 0.3
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging
        if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
            tau_in = 0.3
            tau_out = 6
            tau_open = 120
            tau_close = 80
            v_gate = 0.13
        elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
            k = 8.0
            a = 0.15
            epsilon_0 = 0.002
            mu1 = 0.2
            mu2 = 0.3
    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        heart_model_parameters = {
            'tau_in_voxel': np.ones(n_voxel) * tau_in, # determines the shape of action potential
            'tau_out_voxel': np.ones(n_voxel) * tau_out, # determines the shape of action potential
            'tau_open_voxel': np.ones(n_voxel) * tau_open, # determines the shape of action potential
            'tau_close_voxel': np.ones(n_voxel) * tau_close, # determines the shape of action potential
            'v_gate_voxel': np.ones(n_voxel) * v_gate, # gating variable threshold
            'c_voxel': np.ones(n_voxel) * 0.5, # diffusion coefficient
        }
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
        heart_model_parameters = {
            'k_voxel': np.ones(n_voxel) * k,
            'a_voxel': np.ones(n_voxel) * a,
            'epsilon_0_voxel': np.ones(n_voxel) * epsilon_0,
            'mu1_voxel': np.ones(n_voxel) * mu1,
            'mu2_voxel': np.ones(n_voxel) * mu2,
            'c_voxel': np.ones(n_voxel) * 1.6, # diffusion coefficient
        }

    return heart_model_parameters

def scale_heart_model_time(simulation_parameters, arrhythmia_parameters, heart_model_parameters):
    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer
        simulation_parameters['time_scale'] = 1
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov
        simulation_parameters['time_scale'] = 6
        
    simulation_parameters['t_final'] = simulation_parameters['t_final'] / simulation_parameters['time_scale']
    simulation_parameters['dt'] = simulation_parameters['dt'] / simulation_parameters['time_scale']
    arrhythmia_parameters['pacing_start_time'] = arrhythmia_parameters['pacing_start_time'] / simulation_parameters['time_scale']
    arrhythmia_parameters['pacing_cycle_length'] = arrhythmia_parameters['pacing_cycle_length'] / simulation_parameters['time_scale']
    arrhythmia_parameters['s1_t'] = arrhythmia_parameters['s1_t'] / simulation_parameters['time_scale']
    arrhythmia_parameters['s1_s2_delta_t'] = arrhythmia_parameters['s1_s2_delta_t'] / simulation_parameters['time_scale']
    heart_model_parameters['c_voxel'] = heart_model_parameters['c_voxel'] * simulation_parameters['time_scale']

    return simulation_parameters, arrhythmia_parameters, heart_model_parameters