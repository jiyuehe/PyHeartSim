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

import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import numpy as np
import simulation
import utility

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir
    directory['mesh_npz'] = Path('/home/j/Desktop/hdd/share_folder/carto3_files/data npz')
    directory['mesh_obj'] = Path('/home/j/Desktop/hdd/share_folder/carto3_files/mesh obj')
    directory['data'] = Path('/home/j/Desktop/hdd/share_folder/carto3_files/data npz')
    directory['result'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results')

    # create the folder if it does not exist
    directory['data'].mkdir(exist_ok=True)
    directory['result'].mkdir(exist_ok=True)
    
    return directory

def mesh_name():
    # atrial mesh .obj file name
    name_prefixes = {
        103: '103_5-2-1-1-3-Rp-ReLA CS REF 230', # flutter reentry
        104: '104_2-LA fam', # flutter reentry
        105: '105_3-LA FAM', # flutter reentry
        106: '106_2-LA fam', # flutter reentry
        107: '107_3-LA CL 270', # flutter reentry, best
        109: '109_3-LA FAM', # flutter reentry, most dense
        110: '110_1-LA FAM', # flutter reentry
        111: '111_6-LA', # flutter reentry
        112: '112_6-LA CL 300' # flutter reentry
    }

    return name_prefixes

def assign_simulation_parameters(directory, name_prefix, geometry_data):
    if name_prefix == 'sheet':
        geometry_flag = 0  # 2D
    else:
        geometry_flag = 1  # 3D

    simulation_parameters = {
        'geometry_flag': geometry_flag,
        'compute_electrogram_flag': 1, 
        # 1: compute electrogram 
        # 0: do not compute electrogram
        'save_action_potential_of_all_voxel_flag': 0,
        # 1: save action potential of all voxels
        # 0: only save action potential of electrode voxels
        'voxel_id_of_simulation_electrode': geometry_data['voxel_id_of_simulation_electrode'], # electrode locations for computing electrograms
        't_final': 1000, # ms
        'dt': 0.5, # ms. 0.5 is good. if dt is too large, simulation will become numerically unstable
        'heart_model_flag': 0, # 0: Mitchell-Schaeffer, 1: Aliev-Panfilov
        'arrhythmia_flag': 3,
        # 0: focal (perpetual pacings at one location)
        # 1: rotor (via s1-s2 pacing)
        # 2: fibrillation (starts with a rotor via s1-s2 pacing, then becomes fibrillation)
        # 3: according to node_flag (can generate focal / rotor / macro-reentry flutter)
    }

    s2 = []
    node_flag = []

    if simulation_parameters['arrhythmia_flag'] == 3: # load node_flag for simulation with designed tissue properties
        file_path = directory['result'] / f'{name_prefix}_node_flag.npy'
        vertex_flag = np.load(file_path)
        node = geometry_data['voxel']
        vertex = geometry_data['vertex']
        face = geometry_data['face']
        node_flag = utility.mesh_related.project_vertex_flag_to_node_flag(vertex, face, vertex_flag, node)

        s1 = np.where(node_flag == 1)[0] # s1 pacing voxel id
    else:
        vid = 21950 # can use ui_select_nodes.py to find out the vertex id
        vertex_xyz = geometry_data['vertex'][vid,:]
        node = geometry_data['voxel']
        s1 = np.argmin(np.linalg.norm(node - vertex_xyz, axis=1)) # s1 pacing voxel id

    if simulation_parameters['arrhythmia_flag'] in (1, 2): # rotor or fibrillation
        s2 = simulation.pacing.find_out_s2_pacing_voxel_ids_for_rotor_arrhythmia(s1, geometry_data)

    debug_plot = 0
    if debug_plot == 1: 
        # show pacing voxels
        voxel = geometry_data['voxel']
        traces = [
            go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                name='voxels'
            ),
            go.Scatter3d(
                x=[voxel[s1, 0]], y=[voxel[s1, 1]], z=[voxel[s1, 2]],
                mode='markers',
                marker=dict(size=6, color='blue'),
                name='s1'
            ),
        ]
        if str(s2) != '[]':
            traces.append(go.Scatter3d(
                x=voxel[s2, 0], y=voxel[s2, 1], z=voxel[s2, 2],
                mode='markers',
                marker=dict(size=6, color='red'),
                name='s2'
            ))
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data', dragmode='orbit'),
            legend=dict(itemsizing='constant'),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.show()

    if simulation_parameters['arrhythmia_flag'] in (0,): # focal (perpetual pacings at one location)
        params = dict(pacing_start_time=0, pacing_cycle_length=800, s1_t=0, s1_s2_delta_t=0) # s1_t, s1_s2_delta_t are not used
    elif simulation_parameters['arrhythmia_flag'] in (1,): # rotor (via s1-s2 pacing)
        params = dict(pacing_start_time=0, pacing_cycle_length=0, s1_t=0, s1_s2_delta_t=220) # pacing_cycle_length is not used
    elif simulation_parameters['arrhythmia_flag'] in (2,): # fibriilation
        params = dict(pacing_start_time=0, pacing_cycle_length=0, s1_t=0, s1_s2_delta_t=280) # pacing_cycle_length is not used
    elif simulation_parameters['arrhythmia_flag'] == 3: # according to node_flag
        params = dict(pacing_start_time=0, pacing_cycle_length=0, s1_t=0, s1_s2_delta_t=0) # pacing_cycle_length, s1_t, s1_s2_delta_t are not used
    arrhythmia_parameters = {
        'pacing_start_time': params['pacing_start_time'], # ms
        'pacing_cycle_length': params['pacing_cycle_length'], # ms
        's1_pacing_voxel_id': s1, # node id for s1 pacing
        's2_pacing_voxel_id': s2, # node id for s2 pacing
        's1_t': params['s1_t'], # ms. time of s1 pacing
        's1_s2_delta_t': params['s1_s2_delta_t'], # ms. time interval between s1 and s2
    }

    if simulation_parameters['arrhythmia_flag'] in (0,1,3):
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13) # ms: Mitchell-Schaeffer model parameters
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3) # ap: Aliev-Panfilov model parameters
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        ms = dict(tau_in=0.3,  tau_out=12, tau_open=80, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)

    n_voxel = geometry_data['voxel'].shape[0]
    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        heart_model_parameters = {
            'tau_in_voxel': np.ones(n_voxel) * ms['tau_in'], # determines the shape of action potential up stroke
            'tau_out_voxel': np.ones(n_voxel) * ms['tau_out'], # determines the shape of action potential down stroke
            'tau_open_voxel': np.ones(n_voxel) * ms['tau_open'], # determines the time from end of an action potential to start of next action potential
            'tau_close_voxel': np.ones(n_voxel) * ms['tau_close'], # determines the time of action potential duration
            'v_gate_voxel': np.ones(n_voxel) * ms['v_gate'], # gating variable threshold
            'c_voxel': np.ones(n_voxel) * 0.6, # diffusion coefficient
        }
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
        heart_model_parameters = {
            'k_voxel': np.ones(n_voxel) * ap['k'],
            'a_voxel': np.ones(n_voxel) * ap['a'],
            'epsilon_0_voxel': np.ones(n_voxel) * ap['epsilon_0'],
            'mu1_voxel': np.ones(n_voxel) * ap['mu1'],
            'mu2_voxel': np.ones(n_voxel) * ap['mu2'],
            'c_voxel': np.ones(n_voxel) * 1.6, # diffusion coefficient
        }

    # scale heart model time
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
    arrhythmia_parameters['node_flag'] = node_flag
    heart_model_parameters['c_voxel'] = heart_model_parameters['c_voxel'] * simulation_parameters['time_scale']

    return simulation_parameters, arrhythmia_parameters, heart_model_parameters
