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

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir
    directory['mesh_database'] = script_dir / 'mesh_database'
    directory['result'] = script_dir.parent / 'result'
    directory['data'] = script_dir.parent / 'data'

    # create the folder if it does not exist
    directory['result'].mkdir(exist_ok=True)
    directory['data'].mkdir(exist_ok=True)

    return directory

def mesh_name():
    # atrial mesh .obj file name
    name_prefix = 'sheet'
    # name_prefix = 'long_slab'
    # name_prefix = '103_1-lagood'
    # name_prefix = '102_1-LA FAM1'

    return name_prefix

def assign_simulation_parameters(name_prefix, geometry_data, s1, s2, n_voxel):
    if name_prefix == 'sheet':
        geometry_flag = 0  # 2D
    elif name_prefix == 'long_slab':
        geometry_flag = 2  # long slab for computing conduction velocity
    else:
        geometry_flag = 1  # 3D

    simulation_parameters = {
        'geometry_flag': geometry_flag,
        'compute_electrogram_flag': 1, 
        # 1: compute electrogram 
        # 0: do not compute electrogram
        'save_action_potential_of_all_voxel_flag': 1,
        # 1: save action potential of all voxels
        # 0: only save action potential of electrode voxels
        'voxel_id_of_electrode': geometry_data['voxel_id_of_electrode'], # electrode locations for computing electrograms
        't_final': 1000, # ms
        'dt': 0.5, # ms. 0.5 is good. if dt is too large, simulation will become numerically unstable
        'heart_model_flag': 0, # 0: Mitchell-Schaeffer, 1: Aliev-Panfilov
        'arrhythmia_flag': 1,
        # 0: focal
        # 1: rotor
        # 2: fibrillation
        # 3: for debugging, manually assign s1 and s2 region 
        # 4: 2 focal 2 locations both at 0ms
        # 5: 2 focal 2 locations 300ms apart
        # 6: 2 focal 2 locations 50ms apart
    }

    debug_plot = 0
    if debug_plot == 1: # show geometry voxel
        voxel = geometry_data['voxel']
        n_node = voxel.shape[0]
        voxel_id_of_electrode = simulation_parameters['voxel_id_of_electrode']

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

    # NOTE: changes of heart_model_parameters or pacing magnitude/duration will change the ap/h_min/max thresholds
    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        params = dict(pacing_start_time=10,  pacing_cycle_length=300,
                      s1_t=0,   s1_s2_delta_t=0,
                      ap_min=0, ap_max=0, h_min=0, h_max=0, s2_region_size_factor=0)
    elif simulation_parameters['arrhythmia_flag'] == 1: # simple rotor
        params = dict(pacing_start_time=0,   pacing_cycle_length=0,
                      s1_t=0,   s1_s2_delta_t=220,
                      ap_min=0.00038510505014280766, ap_max=0.07687293043769826,
                      h_min=0.1623103413330824,      h_max=0.39717038588499276,
                      s2_region_size_factor=0.7)
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        params = dict(pacing_start_time=0,   pacing_cycle_length=0,
                      s1_t=0,   s1_s2_delta_t=130,
                      ap_min=0.05,  ap_max=0.25, h_min=0.025, h_max=0.120,
                      s2_region_size_factor=0.5)
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging, manually assign s1 and s2 region
        params = dict(pacing_start_time=0,   pacing_cycle_length=0,
                      s1_t=0,   s1_s2_delta_t=230,
                      ap_min=0, ap_max=0, h_min=0, h_max=0, s2_region_size_factor=0)

    arrhythmia_parameters = {
        'pacing_start_time':   params['pacing_start_time'],  # ms
        'pacing_cycle_length': params['pacing_cycle_length'], # ms
        's1_pacing_voxel_id':  s1,                           # node id for s1 pacing
        's2_pacing_voxel_id':  s2,                           # node id for s2 pacing
        's1_t':                params['s1_t'],                # ms. time of s1 pacing
        's1_s2_delta_t':       params['s1_s2_delta_t'],      # ms. time interval between s1 and s2
        'ap_min':              params['ap_min'],              # a threshold value of action potential
        'ap_max':              params['ap_max'],              # a threshold value of action potential
        'h_min':               params['h_min'],               # a threshold value of action potential
        'h_max':               params['h_max'],               # a threshold value of action potential
        's2_region_size_factor': params['s2_region_size_factor'], # a less than 1 multiplication factor to reduce s2 pacing region size
    }

    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13) # ms: Mitchell-Schaeffer model parameters
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3) # ap: Aliev-Panfilov model parameters
    elif simulation_parameters['arrhythmia_flag'] == 1: # simple rotor
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        ms = dict(tau_in=0.08, tau_out=6, tau_open=80,  tau_close=30, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)

    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        heart_model_parameters = {
            'tau_in_voxel':    np.ones(n_voxel) * ms['tau_in'],    # determines the shape of action potential
            'tau_out_voxel':   np.ones(n_voxel) * ms['tau_out'],   # determines the shape of action potential
            'tau_open_voxel':  np.ones(n_voxel) * ms['tau_open'],  # determines the shape of action potential
            'tau_close_voxel': np.ones(n_voxel) * ms['tau_close'], # determines the shape of action potential
            'v_gate_voxel':    np.ones(n_voxel) * ms['v_gate'],    # gating variable threshold
            'c_voxel':         np.ones(n_voxel) * 0.5,             # diffusion coefficient
        }
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
        heart_model_parameters = {
            'k_voxel':         np.ones(n_voxel) * ap['k'],
            'a_voxel':         np.ones(n_voxel) * ap['a'],
            'epsilon_0_voxel': np.ones(n_voxel) * ap['epsilon_0'],
            'mu1_voxel':       np.ones(n_voxel) * ap['mu1'],
            'mu2_voxel':       np.ones(n_voxel) * ap['mu2'],
            'c_voxel':         np.ones(n_voxel) * 1.6,             # diffusion coefficient
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
    heart_model_parameters['c_voxel'] = heart_model_parameters['c_voxel'] * simulation_parameters['time_scale']

    return simulation_parameters, arrhythmia_parameters, heart_model_parameters
