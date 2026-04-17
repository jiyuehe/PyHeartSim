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

def find_out_s2_pacing_voxel_ids_for_rotor_arrhythmia(s1, geometry_data):
    voxel = geometry_data['voxel']
    n_voxel = voxel.shape[0]

    # find a voxel that is at a certain distance from s1
    d_threshold_1 = 15 # mm
    d_threshold_2 = d_threshold_1 + 20 # mm
    d = np.sqrt(np.sum((voxel - voxel[s1, :])**2, axis=1))
    candidate_s2 = np.where((d >= d_threshold_1) & (d <= d_threshold_2))[0]

    # cluster the candidate_s2 voxels into whatever number of cluster by connectivity
    neighbor_id_2d = geometry_data['neighbor_id_2d']
    visited = np.zeros(n_voxel, dtype=bool)
    clusters = []
    for voxel_id in candidate_s2:
        if not visited[voxel_id]:
            cluster = []
            stack = [voxel_id]
            visited[voxel_id] = True

            while stack:
                current_voxel_id = stack.pop()
                cluster.append(current_voxel_id)

                neighbors = neighbor_id_2d[current_voxel_id, :]
                for neighbor in neighbors:
                    if neighbor in candidate_s2 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            clusters.append(cluster)
    
    # find the largest cluster
    largest_cluster = max(clusters, key=len)
    candidate_s2 = largest_cluster[0] # s2 pacing voxel id

    # if the amount of voxels in the largest cluster is larger than a threshold, select a subset of connected voxels: started from the first voxel, then add neighboring voxels until reaching the threshold, according to breadth first search
    n_threshold = 300
    if len(largest_cluster) > n_threshold:
        visited = np.zeros(n_voxel, dtype=bool)
        s2_pacing_voxel_id = []
        queue = [candidate_s2]
        visited[candidate_s2] = True

        while queue and len(s2_pacing_voxel_id) < n_threshold:
            current_voxel_id = queue.pop(0)
            s2_pacing_voxel_id.append(current_voxel_id)

            neighbors = neighbor_id_2d[current_voxel_id, :]
            for neighbor in neighbors:
                if neighbor in largest_cluster and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        s2 = s2_pacing_voxel_id # s2 pacing voxel id
    else:
        s2 = candidate_s2 # s2 pacing voxel id

    return s2

def assign_pacing_parameters(arrhythmia_parameters, arrhythmia_flag, n_voxel, neighbor_id_2d, simulation_parameters):
    s1_pacing_voxel_id = arrhythmia_parameters['s1_pacing_voxel_id'] 
    s1_t = arrhythmia_parameters['s1_t'] 
    s2_t = s1_t + arrhythmia_parameters['s1_s2_delta_t'] 

    s2_pacing_voxel_id = []
    if arrhythmia_flag in (1, 2, 3, 4, 5, 6):
        s2_pacing_voxel_id = arrhythmia_parameters['s2_pacing_voxel_id'] 

    do_flag = 0
    if do_flag == 1:
        # add neighboring nodes to the s1 pacing location
        if isinstance(s1_pacing_voxel_id, int):
            neighbor_id = neighbor_id_2d[s1_pacing_voxel_id, :] # add all the neighbors of the pacing voxel to be paced
            neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
            if np.isscalar(s1_pacing_voxel_id): # if s1_pacing_voxel_id is just a number
                s1_pacing_voxel_id = np.array([s1_pacing_voxel_id]) # np.concatenate will not work with number, that's why convert it to 1d array so that np.concatenate can work
            s1_pacing_voxel_id = np.concatenate([s1_pacing_voxel_id, neighbor_id])
            s1_pacing_voxel_id = np.unique(s1_pacing_voxel_id)

    # initialize pacing stimulus
    J_stim = np.zeros(n_voxel)

    # it seems the rotor will sustain longer if the magnitude is larger, and/or the pacing duration is longer
    # if magnitude or duration is too small, it may not induce rotor
    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        J_stim_magnitude = 1
        pacing_duration = 5 # ms
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
        J_stim_magnitude = 1
        pacing_duration = 5 / simulation_parameters['time_scale']

    return J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t

def apply_pacing(arrhythmia_parameters, simulation_parameters, arrhythmia_flag, model_time, J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, sim_u_voxel, sim_h_voxel, neighbor_id_2d):
    # s1 pacing
    if arrhythmia_flag in (0, 4): # focal arrhythmia, s1 pace according to cycle length setting
        t = model_time
        while t - arrhythmia_parameters['pacing_start_time'] > arrhythmia_parameters['pacing_cycle_length']:
            t = t - arrhythmia_parameters['pacing_cycle_length']

        if t >= arrhythmia_parameters['pacing_start_time'] and t <= arrhythmia_parameters['pacing_start_time'] + pacing_duration:
            if arrhythmia_flag in (0, 4):
                J_stim[s1_pacing_voxel_id] = J_stim_magnitude
            
            if arrhythmia_flag == 4:
                J_stim[s2_pacing_voxel_id] = J_stim_magnitude

    if arrhythmia_flag == 5:
        f1_time = 0/simulation_parameters['time_scale'] # focal 1 pacing time
        f2_time = 300/simulation_parameters['time_scale'] # focal 2 pacing time
        if model_time >= f1_time and model_time <= f1_time + pacing_duration:
            J_stim[s1_pacing_voxel_id] = J_stim_magnitude
        if model_time >= f2_time and model_time <= f2_time + pacing_duration:
            J_stim[s2_pacing_voxel_id] = J_stim_magnitude
    
    if arrhythmia_flag == 6:
        f1_time = 0/simulation_parameters['time_scale'] # focal 1 pacing time
        f2_time = 50/simulation_parameters['time_scale'] # focal 2 pacing time
        if model_time >= f1_time and model_time <= f1_time + pacing_duration:
            J_stim[s1_pacing_voxel_id] = J_stim_magnitude
        if model_time >= f2_time and model_time <= f2_time + pacing_duration:
            J_stim[s2_pacing_voxel_id] = J_stim_magnitude

    elif arrhythmia_flag in (1, 2, 3): # not focal arrhythmia, s1 pace only once
        if model_time >= s1_t and model_time <= s1_t + pacing_duration:
            J_stim[s1_pacing_voxel_id] = J_stim_magnitude

    # s2 pacing
    if arrhythmia_flag in (1, 2, 3) and model_time >= s2_t and model_time <= s2_t + pacing_duration:
        J_stim[s2_pacing_voxel_id] = J_stim_magnitude

    return J_stim
