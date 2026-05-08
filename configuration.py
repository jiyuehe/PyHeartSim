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

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir
    directory['mesh_database'] = script_dir / 'mesh_database' / 'left_atrium'
    directory['data'] = Path('/home/j/Desktop/hdd/share_folder/patient_data')
    directory['result'] = Path('/home/j/Desktop/hdd/share_folder/simulation_results')

    # create the folder if it does not exist
    directory['result'].mkdir(exist_ok=True)
    directory['data'].mkdir(exist_ok=True)

    return directory

def mesh_name(mesh_id):
    # atrial mesh .obj file name
    
    name_prefixes = {
        0: '0_1-la1 78 240',
        1: '1_1-x',
        2: '2_2-lafam pr',
        3: '3_1-la1',
        4: '4_2-LA FAM',
        5: '5_2-la mem',
        6: '6_2-LA MEM',
        7: '7_2-LA FAM',
        8: '8_3-la fam1',
        9: '9_2-la fam',
        10: '10_2-LA FAM',
        11: '11_2-LA FAM',
        12: '12_2-LA FAM',
        13: '13_2-LA FAM',
        14: '14_2-LA FAM',
        15: '15_2-LA FAM',
        16: '16_2-LA FAM',
        17: '17_2-la fam lat export',
        18: '18_2-LA FAM',
        19: '19_1-x',
        20: '20_1-LA FAM',
        21: '21_1-LA FAM',
        22: '22_2-la voltage',
        23: '23_4-Map',
        24: '24_1-LA FAM',
        25: '25_2-la1',
        26: '26_2-la',
        27: '27_2-LA_FAM',
        28: '28_2-LA_FAM',
        29: '29_2-LA_FAM',
        30: '30_3-la_rf',
        31: '31_2-LA_FAM',
        32: '32_2-LA_FAM',
        33: '33_2-LA_FAM',
        34: '34_3-1-1-la3',
        35: '35_2-la',
        36: '36_4-la2',
        37: '37_1-x',
        38: '38_2-la',
        39: '39_2-LA',
        40: '40_3-1-1-1-1-AT350',
        41: '41_2-1-Rela1',
        42: '42_2-LA',
        43: '43_2-LA',
        44: '44_2-LA',
        45: '45_3-1-LAPOSTPVI',
        46: '46_2-1-ReLA',
        47: '47_3-la',
        48: '48_2-pre rf',
        49: '49_2-LA',
        50: '50_2-LA',
        51: '51_2-LA',
        52: '52_2-LA',
        53: '53_2-la voltage',
        54: '54_2-LA 220',
        55: '55_2-1-ReLA 233',
        56: '56_2-1-1-1-Revoltage pa',
        57: '57_2-1-Rela',
        58: '58_2-LA',
        59: '59_2-LA_1',
        60: '60_2-LA_ACTIVATION',
        61: '61_1-1-1-1-1-1-Rela',
        62: '62_2-1-LA2VOLTAGE',
        63: '63_3-LA2',
        64: '64_2-LA',
        65: '65_3-LA 220',
        66: '66_3-1-1-ReLA_PostMIRoofPVISept',
        67: '67_2-LA',
        68: '68_3-LA',
        69: '69_2-LA',
        70: '70_2-LA_1',
        71: '71_1-LA',
        72: '72_2-LA_POSTPVI',
        73: '73_2-LA_1',
        74: '74_2-LA_1',
        75: '75_2-1-ReLA_FL2',
        76: '76_2-lalat',
        77: '77_2-LA_1',
        78: '78_2-LA_1',
        79: '79_2-LA',
        80: '80_2-LA',
        81: '81_1-LA_d',
        82: '82_1-LA_1',
        83: '83_2-LA',
        84: '84_2-LAVOLTAGE',
        85: '85_2-LA_PreAbl',
        86: '86_2-LA_2',
        87: '87_1-LA_FIB_VOLT',
        88: '88_1-LA',
        89: '89_2-LAFAM FIB',
        90: '90_1-sinus volt map',
        91: '91_2-LA sinus with ablation lesions',
        92: '92_1-la volt',
        93: '93_1-1-ReLA FLUTTER 1',
        94: '94_1-1-ReLA voltage_sinus',
        95: '95_1-SINUS CARTO FINDER',
        96: '96_1-LA PRE',
        97: '97_1-LAFAM_post_finder',
        98: '98_1-LaFAM_study_data',
        99: '99_2-LaFAM_cartofinder_data',
        100: '100_1-LA FAM1',
        101: '101_1-lagood',
        102: '102_2-1-1-ReLA FAM AT',
        103: '103_6-LA',
    }

    name_prefix = name_prefixes[mesh_id]

    return name_prefix

def assign_simulation_parameters(name_prefix, geometry_data, s1, s2):
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
        't_final': 500, # ms
        'dt': 0.5, # ms. 0.5 is good. if dt is too large, simulation will become numerically unstable
        'heart_model_flag': 0, # 0: Mitchell-Schaeffer, 1: Aliev-Panfilov
        'arrhythmia_flag': 4,
        # 0: focal (multiple cycles)
        # 1: rotor
        # 2: fibrillation (starts with a rotor, then becomes fibrillation)
        # 3: for debugging, manually assign s1 and s2 region 
        # 4: 1 focal 1 cycle
        # 5: 2 focal 2 locations 300ms apart
        # 6: 2 focal 2 locations 50ms apart
    }

    # if simulate rotor or fibrillation, s2 will be automatically determined
    if simulation_parameters['arrhythmia_flag'] in (1, 2): # rotor or fibrillation
        s2 = simulation.pacing.find_out_s2_pacing_voxel_ids_for_rotor_arrhythmia(s1, geometry_data)

    # NOTE: changes of heart_model_parameters or pacing magnitude/duration will change the ap/h_min/max thresholds
    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        params = dict(pacing_start_time=10, pacing_cycle_length=300, s1_t=0, s1_s2_delta_t=0)
    elif simulation_parameters['arrhythmia_flag'] in (1, 2): # rotor or fibrillation
        params = dict(pacing_start_time=0, pacing_cycle_length=0, s1_t=0, s1_s2_delta_t=350)
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging, manually assign s1 and s2 region
        params = dict(pacing_start_time=0, pacing_cycle_length=0, s1_t=0, s1_s2_delta_t=230)
    arrhythmia_parameters = {
        'pacing_start_time': params['pacing_start_time'], # ms
        'pacing_cycle_length': params['pacing_cycle_length'], # ms
        's1_pacing_voxel_id': s1, # node id for s1 pacing
        's2_pacing_voxel_id': s2, # node id for s2 pacing
        's1_t': params['s1_t'], # ms. time of s1 pacing
        's1_s2_delta_t': params['s1_s2_delta_t'], # ms. time interval between s1 and s2
    }

    if simulation_parameters['arrhythmia_flag'] in (0, 4, 5, 6): # focal
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13) # ms: Mitchell-Schaeffer model parameters
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3) # ap: Aliev-Panfilov model parameters
    elif simulation_parameters['arrhythmia_flag'] == 1: # simple rotor
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)
    elif simulation_parameters['arrhythmia_flag'] == 2: # fibrillation
        ms = dict(tau_in=0.3,  tau_out=12, tau_open=30, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)
    elif simulation_parameters['arrhythmia_flag'] == 3: # for debugging
        ms = dict(tau_in=0.3,  tau_out=6, tau_open=120, tau_close=80, v_gate=0.13)
        ap = dict(k=8.0, a=0.15, epsilon_0=0.002, mu1=0.2, mu2=0.3)

    n_voxel = geometry_data['voxel'].shape[0]

    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        heart_model_parameters = {
            'tau_in_voxel': np.ones(n_voxel) * ms['tau_in'], # determines the shape of action potential
            'tau_out_voxel': np.ones(n_voxel) * ms['tau_out'], # determines the shape of action potential
            'tau_open_voxel': np.ones(n_voxel) * ms['tau_open'], # determines the shape of action potential
            'tau_close_voxel': np.ones(n_voxel) * ms['tau_close'], # determines the shape of action potential
            'v_gate_voxel': np.ones(n_voxel) * ms['v_gate'], # gating variable threshold
            'c_voxel': np.ones(n_voxel) * 4.0, # diffusion coefficient
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
    heart_model_parameters['c_voxel'] = heart_model_parameters['c_voxel'] * simulation_parameters['time_scale']

    return simulation_parameters, arrhythmia_parameters, heart_model_parameters
