import numpy as np
import matplotlib.pyplot as plt

def execute_on_action_potential(t_start, a_p, v_gate, cycle_length_percentage):    
    # find out cycle length
    threshold = v_gate
    n_signals = a_p.shape[1] # a_p is action potential
    markers = []
    cycle_lengths = []
    for n in range(n_signals):
        marker_temp = find_action_potential_marker(a_p[:,n], threshold)
        if not np.isnan(marker_temp).all():
            cl = np.diff(marker_temp)
            markers.append(marker_temp)
            cycle_lengths.append(cl)
        else:
            markers.append(np.array([np.nan]))
            cycle_lengths.append(np.array([np.nan]))
    
    debug_plot = 0
    if debug_plot == 1: # cycle length histogram
        cl = np.concatenate(cycle_lengths)
        plt.figure()
        plt.hist(cl, bins=10)
        plt.xlabel('cycle length, ms')
        plt.ylabel('counts')
        plt.title('cycle length histogram')
        plt.show()

    # Flatten all cycle lengths
    cl = np.concatenate(cycle_lengths)
    cl = cl[~np.isnan(cl)]
    if len(cl) > 0:
        CL = np.mean(cl)
    else: # this means the action potential time is short, only contains less then 1 cycle of time
        CL = 180 # give a generic value of arrhythmia
    
    # window of interest
    woi = [t_start, t_start + round(CL * cycle_length_percentage)]
    lat = np.full(n_signals, np.nan)
    for n in range(n_signals):
        marker = markers[n]
        marker = marker[~np.isnan(marker)]
        m = marker[(marker >= woi[0]) & (marker <= woi[1])]
        if len(m) > 0:
            lat[n] = m[0]
    
    # shift values so that it starts at 1 ms
    lat = lat - np.nanmin(lat) + 1
    
    return lat, cl # lat.shape = (number of mesh vertices,). cl.shape = (n x number of mesh vertices, ) where n depends on how many cycle length is there in the action potential

def find_action_potential_marker(s, threshold): # m is the activation marker. diff(m) will be the cycle length
    above_threshold_id = np.where(s > threshold)[0]
    
    debug_plot = 0
    if debug_plot == 1:
        plt.figure()
        plt.plot(s, 'k')
        plt.plot(s, '.b')
        plt.plot(above_threshold_id, s[above_threshold_id], '.r')
        plt.show()

    if len(above_threshold_id) > 0:  # there is at least one activation
        b = np.where(np.diff(above_threshold_id) > 1)[0]  # beat separator
        
        debug_plot = 0
        if debug_plot == 1:
            plt.figure()
            plt.plot(s, 'k')
            plt.plot(s, '.b')
            plt.plot(above_threshold_id, s[above_threshold_id], '.r')
            plt.scatter(above_threshold_id[b], s[above_threshold_id[b]], s=100, c='g')

        segment_marker = np.concatenate([
            [above_threshold_id[0]],
            above_threshold_id[b],
            above_threshold_id[b + 1],
            [above_threshold_id[-1]]
        ])
        segment_marker = np.sort(segment_marker)
        segment_marker = np.unique(segment_marker)
        
        # two elements create an interval, if the number of elements is odd, then delete the last element
        if len(segment_marker) % 2 == 1:
            segment_marker = segment_marker[:-1]
        
        debug_plot = 0
        if debug_plot == 1:
            plt.figure()
            plt.plot(s, 'k')
            plt.plot(s, '.b')
            plt.plot(above_threshold_id, s[above_threshold_id], '.r')
            plt.scatter(segment_marker, s[segment_marker], s=100, c='g')

        n_segments = len(segment_marker) // 2 # two elements create an interval
        m = np.zeros(n_segments, dtype=int) # must have dtype=int, because m is used as index
        value = np.zeros(n_segments)
        for i in range(n_segments):
            start_idx = segment_marker[i * 2]
            end_idx = segment_marker[i * 2 + 1]
            segment = s[start_idx:end_idx + 1] # s[id1:id2] includes id1 but excludes id2, thus end_idx + 1 to include end_idx
            ds_dt = np.diff(segment)
            value[i] = np.max(ds_dt)
            id_max = np.argmax(ds_dt)
            m[i] = id_max + start_idx
        
        # sometimes s starts with a portion of the action potential, need to delete marker on such case, only mark at positive ds/dt 
        id_to_delete = value <= 0
        m = m[~id_to_delete]
        
        debug_plot = 0 # m is the maximum positive derivative time instance
        if debug_plot == 1:
            plt.figure()
            plt.plot(s, 'k')
            plt.plot(s, '.b')
            plt.scatter(m, s[m], s=100, c='r')

    else:  # there is no activation
        m = np.array([np.nan])
    
    return m