import numpy as np

def assign_pacing_parameters(arrhythmia_parameters, arrhythmia_flag, n_voxel, neighbor_id_2d, simulation_parameters):
    s1_pacing_voxel_id = arrhythmia_parameters['s1_pacing_voxel_id'] 
    s1_t = arrhythmia_parameters['s1_t'] 
    s2_t = s1_t + arrhythmia_parameters['s1_s2_delta_t'] 
    ap_min = arrhythmia_parameters['ap_min'] 
    ap_max = arrhythmia_parameters['ap_max'] 
    h_min = arrhythmia_parameters['h_min'] 
    h_max = arrhythmia_parameters['h_max'] 
    s2_region_size_factor = arrhythmia_parameters['s2_region_size_factor'] 

    s2_pacing_voxel_id = []
    if arrhythmia_flag in (3, 4, 5, 6):
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

    return J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, ap_min, ap_max, h_min, h_max, s2_region_size_factor

def apply_pacing(arrhythmia_parameters, simulation_parameters, arrhythmia_flag, model_time, J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, ap_min, ap_max, h_min, h_max, s2_region_size_factor, sim_u_voxel, sim_h_voxel, neighbor_id_2d):
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
        if arrhythmia_flag == 1 or arrhythmia_flag == 2:
            action_potential_s2_t = sim_u_voxel[:,int(s2_t)-1] # -1: the current values are not saved yet, so check the previous physical time frame
            h_s2_t = sim_h_voxel[:,int(s2_t)-1] # -1: the current values are not saved yet, so check the previous physical time frame

            id1 = np.where((action_potential_s2_t >= ap_min) & (action_potential_s2_t <= ap_max))[0]
            id2 = np.where((h_s2_t >= h_min) & (h_s2_t <= h_max))[0]
            s2_pacing_voxel_id_auto = np.intersect1d(id1, id2) # these voxels could have a ring-like shape, which cannot generate rotor

            # grab a portion of the shape, so it becomes like a curvy patch (instead of a ring), allow waves to rotate at the edges of the patch
            id = s2_pacing_voxel_id_auto[0] # find one voxel to start, can be any random one
            iter = 0
            while (id.size < s2_pacing_voxel_id_auto.size * s2_region_size_factor or id.size < 1000) and iter <= 50: # repeat several times to include more neighbors
                # NOTE: iter <= 10 is to prevent inifinte while loop that sometimes will happen
                neighbor_id = neighbor_id_2d[id, :] # the neighbors
                neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
                id = np.concatenate([np.atleast_1d(id), np.atleast_1d(neighbor_id)]) # add the neighbors
                id = np.intersect1d(id, s2_pacing_voxel_id_auto) # make sure its within the original shape
                iter = iter + 1
            s2_pacing_voxel_id = id
            # print(s2_pacing_voxel_id)
        elif arrhythmia_flag == 3:
            s2_pacing_voxel_id = s2_pacing_voxel_id 
            # s2_pacing_voxel_id = []

        J_stim[s2_pacing_voxel_id] = J_stim_magnitude

    return J_stim
