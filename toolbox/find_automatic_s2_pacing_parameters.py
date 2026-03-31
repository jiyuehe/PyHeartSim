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
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import codes
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = 'browser'
from sklearn.neighbors import NearestNeighbors # pip install scikit-learn

#%%
# MUST READ: 
# 1. use ui_select_nodes.py to manually set s2 pacing sites, then set "rotor_flag = 1" in simulation to see if that generates rotors. 
#    after trails and errors, will find out a s2 pacing region that generates rotors.
# 2. run a focal simulation to save the u and h values. 
#    NOTE that this step is necessary, because if use the rotor simulation, the u and h values will contain the s2 pacing stimulus, 
#    thus cannot find the correct u and h threshold for identifying a s2 pacing region.
# 3. run this python file to find out the u and h threshold.

data_flag = 1 # 1: from PyHeartSim. 2: from SPH-HeartSim
geometry_file_name = '49_2-LA_edited.obj' 
simulation_results_file_name = 'simulation_results_8556.npz'

# ==============================
# parameters for finding the s2 pacing region
if data_flag == 1:
    s1_t =  0
    s2_t = s1_t + 230 
if data_flag == 2:
    s1_t =  0
    s2_t = s1_t + 50 
    s2_t = np.argmin(np.abs(t-s2_t)) # in SPHinXsys, the simulation time intervals are adaptive, thus need to find out the index for the time

# parameters for finding the s2 pacing region
s2_region_size_factor = 0.5 # a less than 1 multiplication factor to reduce s2 pacing region size
# ==============================

# load data
if data_flag == 1:
    data_dir = script_dir.parent / 'data'
    result_dir = script_dir.parent / 'result'

    loaded = np.load(str(data_dir / geometry_file_name)[0:-4] + '.npz')
    geometry_data = {
        'voxel': loaded['voxel'],
        'neighbor_id_2d': loaded['neighbor_id_2d'],
        'Delta': loaded['Delta'],
        'voxel_for_each_vertex': loaded['voxel_for_each_vertex'],
        'vertex_for_each_voxel': loaded['vertex_for_each_voxel'],
        'vertex': loaded['vertex'],
        'face': loaded['face'],
    }
    neighbor_id_2d = geometry_data['neighbor_id_2d']
    node = geometry_data['voxel']
    node_vertex = node[geometry_data['voxel_for_each_vertex'],:]
    n_voxel = geometry_data['voxel'].shape[0]

    simulation_results = np.load(result_dir / simulation_results_file_name)
    action_potential = simulation_results['action_potential']
    h = simulation_results['h']

    node_flag = np.load(data_dir / 'node_flag.npy')
    # node_flag = node_flag[geometry_data['vertex_for_each_voxel']] # node_flag was on vertex, now map it to voxels
elif data_flag == 2:
    data_dir = script_dir.parent / 'SPH-HeartSim' / 'build' / 'sim' / 'bin' / 'output'
    node_flag_dir = script_dir.parent / 'SPH-HeartSim' / 'result'

    t, voltage, gate_variable, stress, xyz = codes.load_SPH_simulation_result.execute(data_dir)
    node = xyz[:,0,:]

    action_potential = voltage
    h = gate_variable

    node_flag = np.load(node_flag_dir / 'node_flag.npy')

#%%
# load the rotor simulation figured out by manual trials and errors
# the manually assigned pacing sites
s1_pacing_node_id = np.where(node_flag == 1)[0]
s2_pacing_node_id = np.where(node_flag == 2)[0]

debug_flag = 0
if debug_flag == 1:
    if s1_pacing_node_id.size > 1:
        print('s1_pacing_node_id:')
        print(','.join(map(str, s1_pacing_node_id)))
    print('s2_pacing_node_id:')
    print(','.join(map(str, s2_pacing_node_id)))

# analyze the successful s2 region
action_potential_s2 = action_potential[s2_pacing_node_id,s2_t]
h_s2 = h[s2_pacing_node_id,s2_t]

print(f"action potential min max: {np.min(action_potential_s2)} {np.max(action_potential_s2)}\n"
    f"h min max: {np.min(h_s2)} {np.max(h_s2)}")

# ==============================
# parameters for finding the s2 pacing region
if data_flag == 1: # for PyHeartSim
    ap_min = 0.00038510505014280766 # reference to np.min(action_potential_s2)
    ap_max = 0.07687293043769826 # reference to np.max(action_potential_s2
    h_min = 0.1623103413330824 # reference to np.min(h_s2)
    h_max = 0.39717038588499276 # reference to np.max(h_s2)
elif data_flag == 2: # for SPH-HeartSim
    ap_min = 0.016783728 # reference to np.min(action_potential_s2)
    ap_max = 0.645235435 # reference to np.max(action_potential_s2
    h_min = 0.431867775 # reference to np.min(h_s2)
    h_max = 2.380168937 # reference to np.max(h_s2)
# ==============================

# plot the values of action_potential_s2 and h_s2
do_flag = 0
if do_flag == 1:
    plt.figure()
    plt.plot(action_potential_s2,'b')
    plt.plot(h_s2,'g')
    plt.xlabel('nodes')
    plt.ylabel('')
    plt.title('b: action potential, g: h')

# plot the manually assigned pacing sites
node_s1 = node[s1_pacing_node_id, :]
node_s2 = node[s2_pacing_node_id, :]
codes.display_s1s2_pacing_sites.execute(node, node_s1, node_s2)

#%% automatically find s2 pacing voxels
# --------------------------------------------------
action_potential_all_nodes = action_potential[:,s2_t]
h_all_nodes = h[:,s2_t]

id1 = np.where((action_potential_all_nodes >= ap_min) & (action_potential_all_nodes <= ap_max))[0]
id2 = np.where((h_all_nodes >= h_min) & (h_all_nodes <= h_max))[0]
s2_pacing_node_id_auto = np.intersect1d(id1, id2) # these voxels have a shape like a ring

# plot the automatically assigned pacing sites
node_s1 = node[s1_pacing_node_id, :]
node_s2 = node[s2_pacing_node_id_auto, :]
codes.display_s1s2_pacing_sites.execute(node_vertex, node_s1, node_s2)
# NOTE: this s2 pacing region could shaped like a ring, cannot generate rotor

#%% 
if data_flag == 2: # compute neighbor id list
    n_neighbors = 6
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(node)
    distances, indices = nbrs.kneighbors(node)
    neighbor_id_2d = indices[:, 1:]  # exclude self (first index)

    '''
    # the method below is slow
    n_neighbors = 6
    n_nodes = len(node)
    neighbor_id_2d = []
    for i in range(n_nodes):
        # compute distance from node i to all other nodes
        diffs = node - node[i] # shape (n_nodes, 3)
        dists = np.sqrt(np.sum(diffs**2, axis=1)) # Euclidean distance
        
        # get indices of the 6 smallest distances (excluding itself)
        nearest_ids = np.argsort(dists)[1:n_neighbors + 1] # skip index 0 (itself)
        neighbor_id_2d.append(nearest_ids)
    neighbor_id_2d = np.array(neighbor_id_2d)
    '''

# grab a portion of the s2 pacing sites, so that it's like a curvy line instead of a ring
id = s2_pacing_node_id_auto[0] # find one voxel to start
while id.size < s2_pacing_node_id_auto.size * s2_region_size_factor: # repeat several times to include more neighbors
    neighbor_id = neighbor_id_2d[id, :] # add all the neighbors of the pacing voxel to be paced
    neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
    id = np.concatenate([np.atleast_1d(id), np.atleast_1d(neighbor_id)]) # add the neighbors
    id = np.intersect1d(id, s2_pacing_node_id_auto) # make sure its within the original shape
s2_pacing_node_id = id

# plot the automatically assigned pacing sites
node_s1 = node[s1_pacing_node_id, :]
node_s2 = node[s2_pacing_node_id, :]
codes.display_s1s2_pacing_sites.execute(node_vertex, node_s1, node_s2)

debug_flag = 0
if debug_flag == 1:
    # print the pacing sites
    if s1_pacing_node_id.size > 1:
        print('s1_pacing_node_id:')
        print(','.join(map(str, s1_pacing_node_id)))
    print('s2_pacing_node_id:')
    print(','.join(map(str, s2_pacing_node_id)))

plt.show()

#%%
