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
import time
import numpy as np # pip install numpy
from numba import cuda # pip install numba
import matplotlib.pyplot as plt # pip install matplotlib
import simulation
import utility
from tool import ui_movie
import common
import configuration
from pathlib import Path

import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

#%%
def run_simulation(input_arguments):
    result_folder = input_arguments['result_folder']
    geometry_data = input_arguments['geometry_data']
    simulation_parameters = input_arguments['simulation_parameters']
    arrhythmia_parameters = input_arguments['arrhythmia_parameters']
    heart_model_parameters = input_arguments['heart_model_parameters']

    n_voxel = geometry_data['voxel'].shape[0]

    # fiber orientations
    D0 = simulation.equation_parts.load_fiber(n_voxel)

    # heart model equation parts
    P_2d = simulation.equation_parts.heart_model_equation_parts(simulation_parameters, n_voxel, D0, geometry_data['neighbor_id_2d'], heart_model_parameters)

    # solve differential equations
    start = time.time()
    if cuda.is_available(): # GPU parallel
        print("GPU parallel computing for simulation")
        action_potential, h, physical_time = simulation.simulation_gpu.compute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters)
    end = time.time()
    print(f'simulation completed in {end - start:.1f} seconds')

    # compute unipolar electrogram
    start = time.time()
    if simulation_parameters['compute_electrogram_flag'] == 1:
        voxel = geometry_data['voxel']
        voxel_id_of_simulation_electrode = simulation_parameters['voxel_id_of_simulation_electrode']
        electrode_xyz = voxel[voxel_id_of_simulation_electrode, :]
        Delta = geometry_data['Delta']
        neighbor_id_2d = geometry_data['neighbor_id_2d']
        
        # check GPU availability and choose appropriate module
        if cuda.is_available():
            print("GPU parallel computing for electrogram")
            electrogram_unipolar = simulation.unipolar_electrogram_gpu.compute(electrode_xyz, voxel, D0, heart_model_parameters['c_voxel'], action_potential, Delta, neighbor_id_2d)
    end = time.time()
    print(f'electrogram computation completed in {end - start:.1f} seconds')

    # save simulation results
    voxel_id_of_simulation_electrode = geometry_data['voxel_id_of_simulation_electrode']
    s1 = arrhythmia_parameters['s1_pacing_voxel_id']

    simulation_results = {}
    simulation_results['geometry_flag'] = simulation_parameters['geometry_flag'] # 0: 2D, 1: 3D

    simulation_results['action_potential_electrode'] = action_potential[:, voxel_id_of_simulation_electrode] # shape: (time, n_electrode)

    if simulation_parameters['compute_electrogram_flag'] == 1:
        simulation_results['electrogram_unipolar'] = electrogram_unipolar

    if simulation_parameters['save_action_potential_of_all_voxel_flag'] == 1:
        simulation_results['resolution'] = '1mm'
        simulation_results['action_potential'] = action_potential # shape: (time, n_voxel)
        simulation_results['h'] = h # shape: (time, n_voxel)
    else: 
        simulation_results['resolution'] = '3mm'

    simulation_results['physical_time'] = physical_time

    # save simulation results
    name_prefix = input_arguments['name_prefix']
    name_suffix = input_arguments['s1'][0] if isinstance(input_arguments['s1'], (list, np.ndarray)) else input_arguments['s1']
    np.savez(result_folder / f'{name_prefix}_simulation_results_{name_suffix}', **simulation_results)

#%%
# If running this script directly, the following code block will be executed. 
# If calling the run_simulation() function from another script, the following code block will be ignored.
if __name__ == "__main__":
    directory = configuration.directory_setup() # set up directories
    name_prefixs = configuration.mesh_name()
    name_prefix = name_prefixs[112]

    plot_lat_map_flag = 1 # 1: plot local activation time map. 0: do not plot local activation time map
    plot_phase_map_flag = 1 # 1: plot phase map. 0: do not plot phase map

    # load geometry data
    file_path = directory['data'] / f'{name_prefix}_mesh.npz'
    data = np.load(file_path, allow_pickle=False)
    geometry_data = {k: data[k] for k in data.files}

    # simulation parameter settings
    simulation_parameters, arrhythmia_parameters, heart_model_parameters = configuration.assign_simulation_parameters(directory, name_prefix, geometry_data)

    input_arguments = {}
    input_arguments['name_prefix'] = name_prefix
    input_arguments['geometry_data'] = geometry_data
    input_arguments['result_folder'] = Path('/home/j/Desktop/ssd/git/PyHeartSim/result')
    input_arguments['s1'] = arrhythmia_parameters['s1_pacing_voxel_id']
    input_arguments['s2'] = arrhythmia_parameters['s2_pacing_voxel_id']
    input_arguments['simulation_parameters'] = simulation_parameters
    input_arguments['arrhythmia_parameters'] = arrhythmia_parameters
    input_arguments['heart_model_parameters'] = heart_model_parameters

    # run simulation
    run_simulation(input_arguments)

    name_prefix = input_arguments['name_prefix']
    name_suffix = input_arguments['s1'][0] if isinstance(input_arguments['s1'], (list, np.ndarray)) else input_arguments['s1']
    file_name =  f'{name_prefix}_simulation_results_{name_suffix}.npz'

    # plot some action potentials and electrograms
    do_flag = 1
    if do_flag == 1: 
        # load simulation results
        simulation_results = dict(np.load(input_arguments['result_folder'] / file_name, allow_pickle=False))

        ap = simulation_results['action_potential_electrode']
        if simulation_parameters['compute_electrogram_flag'] == 1:
            egm = simulation_results['electrogram_unipolar']
        elif simulation_parameters['compute_electrogram_flag'] == 0:
            egm = []
        
        n_e_id = 5
        e_id = np.linspace(0, ap.shape[1] - 1, n_e_id, dtype=int)

        fig, axes = plt.subplots(
            nrows=n_e_id, ncols=2, figsize=(6, 8), sharex='col', sharey=False
        )

        for i, eid in enumerate(e_id):
            # left column: action potentials
            axes[i, 0].plot(ap[:, eid])
            # right column: electrograms
            if egm.size > 0:
                axes[i, 1].plot(egm[:, eid])

        plt.tight_layout()
        plt.savefig(input_arguments['result_folder'] / f'{name_prefix}_ap_egm.png', dpi=300)
        plt.close()

    # local activation time map
    if plot_lat_map_flag == 1:
        # compute local activation time
        simulation_results = dict(np.load(input_arguments['result_folder'] / file_name, allow_pickle=False)) # load simulation results
        electrogram_unipolar = simulation_results['electrogram_unipolar'][0:500,:] # shape: (time, n_electrode)
        lat_electrode = utility.lat_map.compute_electrode_lat(electrogram_unipolar)

        voxel_electrode = geometry_data['voxel3mm_1mm_spacing']

        # plot
        fig_name = input_arguments['result_folder'] / f'{name_prefix}_lat.png'
        geometry_flag = simulation_results['geometry_flag']
        utility.lat_map.plot(voxel_electrode, lat_electrode, geometry_flag, fig_name)
        common.crop_image(fig_name)

        # save lat to simulation_results
        simulation_results['lat_electrode'] = lat_electrode
        np.savez(input_arguments['result_folder'] / file_name, **simulation_results)

        # compute conduction velocity
        conduction_velocity_vectors, conduction_velocity_magnitudes, conduction_velocity_mean = utility.conduction_velocity.compute(lat_electrode, geometry_data)
        print(f'mean conduction velocity: {conduction_velocity_mean:.2f} mm/ms')

    # phase map
    if plot_phase_map_flag == 1:
        ap_electrode = simulation_results['action_potential_electrode']  # shape: (time, n_voxel)
        n_electrode = ap_electrode.shape[1]
        ap_phase = []
        activation_phase = []
        v_gate = heart_model_parameters['v_gate_voxel'][geometry_data['voxel_id_of_simulation_electrode']]
        for i in range(n_electrode):
            ap = ap_electrode[:, i]
            ap_phase_i, activation_phase_i = utility.compute_phase.via_action_potential(ap, v_gate[i])
            ap_phase.append(ap_phase_i)
            activation_phase.append(activation_phase_i)

        phase_normalized = np.array(activation_phase).T # shape: (time, n_electrode)

        debug_plot = 0
        if debug_plot == 1:
            plt.figure()
            plt.plot(ap_electrode[:, 0], 'b')
            plt.plot(phase_normalized[:, 0], 'g')
            plt.tight_layout()
            plt.show()

        # phase at some time instance
        t_idx = phase_normalized.shape[0] // 2
        phase_at_t = phase_normalized[t_idx, :]

        geometry_flag_val = int(simulation_results['geometry_flag'])
        color = common.convert_value_to_red_blue(phase_at_t, 0.0, 1.0, -0.1)

        # interactive plotly phase map (cubes)
        cmap = plt.cm.hsv
        rgba = cmap(phase_at_t)  # shape: (n_electrode, 4)
        face_colors_per_cube = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in rgba]

        cube_verts = np.array([
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        ])
        cube_faces = np.array([
            [0,1,2],[0,2,3], [4,5,6],[4,6,7],
            [0,1,5],[0,5,4], [2,3,7],[2,7,6],
            [1,2,6],[1,6,5], [3,0,4],[3,4,7],
        ])  # 12 triangles per cube

        all_x, all_y, all_z, all_i, all_j, all_k, all_fc = [], [], [], [], [], [], []
        for idx, (center, fc) in enumerate(zip(voxel_electrode, face_colors_per_cube)):
            verts = center + cube_verts
            base = 8 * idx
            all_x.extend(verts[:, 0]); all_y.extend(verts[:, 1]); all_z.extend(verts[:, 2])
            for f in cube_faces:
                all_i.append(base + f[0]); all_j.append(base + f[1]); all_k.append(base + f[2])
                all_fc.append(fc)

        fig_plotly = go.Figure(data=go.Mesh3d(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            facecolor=all_fc,
            flatshading=True,
            showscale=False,
        ))
        fig_plotly.update_layout(
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode='data'),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig_plotly.show()

    # display simulation movie
    ui_movie.run_viewer(
        mesh_path=directory['data'] / f'{name_prefix}_mesh.npz',
        result_path=input_arguments['result_folder'] / file_name,
    )

    print('done')
#%%
