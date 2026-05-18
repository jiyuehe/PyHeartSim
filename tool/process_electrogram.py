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

# add the workspace root to Python path
import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

import sys
workspace_root = Path().resolve().parent # Path().resolve() returns an absolute path, the full path
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
import common

directory = {}
directory['home'] = script_dir
directory['data_carto'] = Path('/home/j/Desktop/hdd/share_folder/carto3_files/npz')
directory['data_geometry'] = Path('/home/j/Desktop/hdd/share_folder/patient_data')
directory['result'] = script_dir / 'result'
directory['result'].mkdir(exist_ok=True)

#%%
# setting
half_window_size = 500//2 # number of time points before and after the 2000 ms mark

# load data
name_prefix = '78_2-LA_1'

data = np.load(directory['data_carto'] / f'{name_prefix}_carto_data.npz', allow_pickle=True)
carto_data = {k: data[k] for k in data.files}

data = np.load(directory['data_geometry'] / f'{name_prefix}_clinical_data.npz', allow_pickle=True)
geometry_data = {k: data[k] for k in data.files}

# grab the electrode positions and electrograms
electrode = carto_data['electrode']
electrode_positions_all = carto_data['electrode_positions']

electrode_positions = []
electrogram_unipolar_original = []
electrogram_bipolar_original = []
electrogram_reference_original = []
reference_channel_name = []
for e_id in range(len(electrode)):
    electrode_name = electrode[e_id]['unipolar_name']

    if 'mapping_' in electrode_name:
        # grab full-length electrograms
        uni = electrode[e_id]['unipolar']
        bi = electrode[e_id]['bipolar']

        if uni is not None and bi is not None:
            electrode_positions.append(electrode_positions_all[e_id, :])

            # ref = electrode[e_id]['reference']
            ref = electrode[e_id]['surface'][:,1] # surface lead V1

            electrogram_unipolar_original.append(uni)
            electrogram_bipolar_original.append(bi)
            electrogram_reference_original.append(ref)

            reference_channel_name.append(carto_data['electrode_point_info'][e_id]['reference_channel_name'])
            # reference_channel_name.append('V1')

electrode_positions = np.array(electrode_positions)
electrogram_unipolar_original = np.array(electrogram_unipolar_original)
electrogram_bipolar_original = np.array(electrogram_bipolar_original)
electrogram_reference_original = np.array(electrogram_reference_original)

debug_plot = 0
if debug_plot == 1:
    # plot electrograms of an electrode
    e_id = 301
    plt.figure(figsize=(12, 6))
    plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
    plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
    plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
    plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
    plt.xlabel('ms')
    plt.ylabel('mV')
    plt.legend()
    plt.tight_layout()
    plt.show()

# for each electrode position, find the closest voxel3mm
voxel3mm = geometry_data['voxel3mm']
voxel3mm_id_of_electrode = []
n_electrodes = len(electrode_positions)
max_progress_prints = 10
progress_step = max(1, int(np.ceil(n_electrodes / max_progress_prints)))
for e_idx, e_pos in enumerate(electrode_positions, start=1):
    if (e_idx % progress_step == 0) or (e_idx == n_electrodes):
        print(f'finding closest voxel for electrode {e_idx}/{n_electrodes}')
    distances = np.linalg.norm(voxel3mm - e_pos, axis=1)
    closest_voxel_index = np.argmin(distances)
    voxel3mm_id_of_electrode.append(closest_voxel_index)
voxel3mm_id_of_electrode = np.array(voxel3mm_id_of_electrode)

# plot the electrode positions and the closest atrium nodes
debug_plot = 0
if debug_plot == 1:
    # voxel = geometry_data['voxel']
    voxel3mm = geometry_data['voxel3mm']
    electrode_voxel = voxel3mm[voxel3mm_id_of_electrode, :]
    fig = go.Figure()
    # voxel
    fig.add_trace(go.Scatter3d(x=voxel3mm[:, 0], y=voxel3mm[:, 1], z=voxel3mm[:, 2], mode='markers', marker=dict(size=1, color='grey'), name='voxel'))
    # electrode
    fig.add_trace(go.Scatter3d(x=electrode_positions[:, 0], y=electrode_positions[:, 1], z=electrode_positions[:, 2], mode='markers', marker=dict(size=1, color='red'), name='electrode'))
    # electrode voxel
    fig.add_trace(go.Scatter3d(x=electrode_voxel[:, 0], y=electrode_voxel[:, 1], z=electrode_voxel[:, 2], mode='markers', marker=dict(size=5, color='blue'), name='closest voxel'))
    no_axis = dict(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    fig.update_layout(scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis), title='Electrode positions and closest voxels')
    fig.show()

#%%
# mask the electrograms to the window of interest
t_start = 2000-1 - half_window_size # window of interest start time index
t_end = 2000-1 + half_window_size # window of interest end time index

taper_length = 50 # number of time points for gradual onset/offset at the window edges
taper_sigma = taper_length / 3 # sigma so the ramp reaches ~1% at the edge
taper = np.exp(-0.5 * ((np.arange(taper_length) - taper_length) / taper_sigma) ** 2) # Gaussian ramp from ~0 to 1
woi_window = np.zeros(electrogram_unipolar_original.shape[1])
woi_window[t_start:t_end] = 1.0
woi_window[t_start:t_start + taper_length] = taper
woi_window[t_end - taper_length:t_end] = taper[::-1]

electrogram_unipolar_masked = electrogram_unipolar_original * woi_window
electrogram_bipolar_masked = electrogram_bipolar_original * woi_window

# find QRS timing from the reference electrogram
s = np.abs(electrogram_reference_original)
s[:, :t_start] = 0
s[:, t_end:] = 0
qrs_time = [find_peaks(s[i, :], height=0.7*np.max(s[i, :]), distance=50)[0][0] for i in range(s.shape[0])] # find peaks in the derivative of each reference electrode

# mask out the QRS in the electrograms via inverse flat-top Gaussian window
qrs_taper_size = 50 # number of time points for the Gaussian ramp on each side
qrs_flat_size = 50  # number of time points held at zero in the flat middle region
qrs_taper_sigma = qrs_taper_size / 3 # sigma so the ramp reaches ~1% at the edge
n_electrodes = electrogram_unipolar_masked.shape[0]
n_sig = electrogram_unipolar_masked.shape[1]
qrs_taper_up = np.exp(-0.5 * ((np.arange(qrs_taper_size) - qrs_taper_size) / qrs_taper_sigma) ** 2) # ~0 -> 1
electrogram_unipolar = np.zeros_like(electrogram_unipolar_masked)
electrogram_bipolar = np.zeros_like(electrogram_bipolar_masked)
for i in range(n_electrodes):
    qrs_bump = np.zeros(n_sig)
    qrs_start = qrs_time[i] - qrs_flat_size // 2 - qrs_taper_size
    qrs_flat_start = qrs_start + qrs_taper_size
    qrs_flat_end = qrs_flat_start + qrs_flat_size
    qrs_end = qrs_flat_end + qrs_taper_size
    qrs_bump[qrs_start:qrs_flat_start] = qrs_taper_up
    qrs_bump[qrs_flat_start:qrs_flat_end] = 1.0
    qrs_bump[qrs_flat_end:qrs_end] = qrs_taper_up[::-1]                                # 1 -> ~0
    qrs_window = 1 - qrs_bump # inverse: 1 outside QRS, flat zero at centre, smooth Gaussian tapers
    electrogram_unipolar[i, :] = electrogram_unipolar_masked[i, :] * qrs_window
    electrogram_bipolar[i, :] = electrogram_bipolar_masked[i, :] * qrs_window

debug_plot = 0
if debug_plot == 1:
    e_id = 301

    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(electrogram_reference_original[e_id, :], color = 'cyan', label='Reference Electrogram (original)')
    plt.plot(electrogram_bipolar_original[e_id, :], color = 'magenta', label='Bipolar Electrogram (original)')
    plt.plot(electrogram_unipolar_original[e_id, :], color = 'blue', label='Unipolar Electrogram (original)')
    plt.title('Original. Blue: unipolar, Magenta: bipolar, Cyan: reference')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(5, 1, 2)
    plt.plot(woi_window, color = 'blue')
    plt.title('Window of Interest')
    plt.xlabel('Time Points')
    plt.ylabel('Weight')

    plt.subplot(5, 1, 3)
    plt.plot(electrogram_bipolar_masked[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
    plt.plot(electrogram_unipolar_masked[e_id, :], color = 'blue', label='Unipolar Electro gram (masked)')
    plt.axvline(qrs_time[e_id], color='red', linestyle='--', label='QRS Timing')
    plt.title('Masked to window of interest. Blue: unipolar, Magenta: bipolar, Red dashed line: QRS timing')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.subplot(5, 1, 4)
    plt.plot(qrs_window, color = 'blue')
    plt.title('Window for QRS Masking')
    plt.xlabel('Time Points')
    plt.ylabel('Weight')

    plt.subplot(5, 1, 5)
    plt.plot(electrogram_bipolar[e_id, :], color = 'magenta', label='Bipolar Electrogram (masked)')
    plt.plot(electrogram_unipolar[e_id, :], color = 'blue', label='Unipolar Electrogram (masked)')
    plt.title('QRS removed. Blue: unipolar, Magenta: bipolar')
    plt.xlabel('ms')
    plt.ylabel('mV')

    plt.tight_layout()
    plt.savefig(directory['result'] / f'{name_prefix}_QRS_removal.png', dpi=300) # save as png
    plt.close()

#%%
# assign electrogram to each voxel3mm
clinical_electrogram_unipolar_original = np.zeros((voxel3mm.shape[0], electrogram_unipolar.shape[1]))
clinical_electrogram_bipolar_original = np.zeros((voxel3mm.shape[0], electrogram_bipolar.shape[1]))
clinical_electrogram_unipolar_refined = np.zeros((voxel3mm.shape[0], electrogram_unipolar.shape[1]))
clinical_electrogram_bipolar_refined = np.zeros((voxel3mm.shape[0], electrogram_bipolar.shape[1]))
clinical_electrogram_reference = np.zeros((voxel3mm.shape[0], electrogram_unipolar_original.shape[1]))
for voxel_id in range(voxel3mm.shape[0]):
    # find electrodes mapped to this voxel
    indices = np.where(voxel3mm_id_of_electrode == voxel_id)[0]

    if len(indices) == 1:
        # if only one electrode mapped to this voxel, use its electrogram
        idx = indices[0]
        clinical_electrogram_unipolar_original[voxel_id, :] = electrogram_unipolar_original[idx, :]
        clinical_electrogram_bipolar_original[voxel_id, :] = electrogram_bipolar_original[idx, :]
        clinical_electrogram_unipolar_refined[voxel_id, :] = electrogram_unipolar[idx, :]
        clinical_electrogram_bipolar_refined[voxel_id, :] = electrogram_bipolar[idx, :]
        clinical_electrogram_reference[voxel_id, :] = electrogram_reference_original[idx, :]
    elif len(indices) > 1:
        # if multiple electrodes mapped to this voxel, use the one with the largest bipolar peak-to-peak amplitude
        peak_to_peak_amplitudes = np.ptp(electrogram_bipolar_original[indices, :], axis=1) # peak-to-peak amplitude for each electrode mapped to this voxel
        max_index = np.argmax(peak_to_peak_amplitudes)
        clinical_electrogram_unipolar_original[voxel_id, :] = electrogram_unipolar_original[indices[max_index], :]
        clinical_electrogram_bipolar_original[voxel_id, :] = electrogram_bipolar_original[indices[max_index], :]
        clinical_electrogram_unipolar_refined[voxel_id, :] = electrogram_unipolar[indices[max_index], :]
        clinical_electrogram_bipolar_refined[voxel_id, :] = electrogram_bipolar[indices[max_index], :]
        clinical_electrogram_reference[voxel_id, :] = electrogram_reference_original[indices[max_index], :]

#%%
# sometimes the electrogram has high frequency noise such as 60 Hz noise from power supply etc, apply a moving average smoothing to remove them
window_size = 5 # number of time points in the moving average window
clinical_electrogram_unipolar_refined = np.convolve(clinical_electrogram_unipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_unipolar_refined.shape)
clinical_electrogram_bipolar_refined = np.convolve(clinical_electrogram_bipolar_refined.flatten(), np.ones(window_size)/window_size, mode='same').reshape(clinical_electrogram_bipolar_refined.shape)

# activation time detection
negative_dvdt = -np.diff(clinical_electrogram_unipolar_refined, axis=1, prepend=clinical_electrogram_unipolar_refined[:, [0]])
negative_dvdt_woi = negative_dvdt.copy()
for i in range(clinical_electrogram_unipolar_refined.shape[0]):
    negative_dvdt_woi[i, :t_start] = 0
    negative_dvdt_woi[i, t_end:] = 0

activation_uni = np.zeros(negative_dvdt_woi.shape[0], dtype=int)
for i in range(negative_dvdt_woi.shape[0]):
    peaks, props = find_peaks(negative_dvdt_woi[i, :], height=0.3*np.max(negative_dvdt_woi[i, :]), distance=20)

    if len(peaks) == 0:
        activation_uni[i] = 0
    elif len(peaks) == 1:
        activation_uni[i] = peaks[0]
    else:
        heights = props['peak_heights']
        top2_order = np.argsort(heights)[-2:]  # indices into peaks/heights of 2 largest
        top2_peaks = peaks[top2_order]
        top2_heights = heights[top2_order]

        # sort descending by height
        desc = np.argsort(top2_heights)[::-1]
        top2_peaks = top2_peaks[desc]
        top2_heights = top2_heights[desc]

        # if 2nd largest is not too smaller than the largest, pick the earlier (smaller index) peak
        if top2_heights[1] >= 0.3 * top2_heights[0]:
            activation_uni[i] = min(top2_peaks)
        else:
            activation_uni[i] = top2_peaks[0]

debug_plot = 0
if debug_plot == 1:
    e_id = 301

    plt.figure(figsize=(12, 6))
    plt.plot(clinical_electrogram_unipolar_refined[e_id, :], color='blue', label='Unipolar Electrogram')
    plt.plot(negative_dvdt_woi[e_id, :], color='orange', label='Negative dV/dt')
    plt.scatter(activation_uni[e_id], negative_dvdt_woi[e_id, activation_uni[e_id]], color='red', label='Detected Activation Time')
    plt.title('Activation Time Detection from Unipolar Electrogram')
    plt.xlabel('ms')
    plt.ylabel('mV / mV/ms')
    plt.legend()
    plt.tight_layout()
    plt.show()

absolute_dvdt = np.abs(np.diff(clinical_electrogram_bipolar_refined, axis=1, prepend=clinical_electrogram_bipolar_refined[:, [0]]))
absolute_dvdt_woi = absolute_dvdt.copy()
for i in range(clinical_electrogram_bipolar_refined.shape[0]):
    absolute_dvdt_woi[i, :t_start] = 0
    absolute_dvdt_woi[i, t_end:] = 0
activation_bi = np.argmax(absolute_dvdt_woi, axis=1)

# remove activation time if it's the 1st or last point
for i in range(clinical_electrogram_unipolar_refined.shape[0]):
    if activation_uni[i] == t_start or activation_uni[i] == t_end:
        activation_uni[i] = 0
    if activation_bi[i] == t_start or activation_bi[i] == t_end:
        activation_bi[i] = 0

# remove activation time if the signal is very small
do_flag = 1
if do_flag == 1:
    for e_id in range(clinical_electrogram_unipolar_refined.shape[0]):
        if np.max(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_unipolar_refined[e_id, t_start:t_end]) < 0.2: # > 1 mV is normal. < 0.5 mV is considered dense scar for unipolar
            activation_uni[e_id] = 0
        if np.max(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) - np.min(clinical_electrogram_bipolar_refined[e_id, t_start:t_end]) < 0.2: # > 0.5 mV is normal. < 0.2 mV is considered dense scar for bipolar
            activation_bi[e_id] = 0
            activation_uni[e_id] = 0 # if bipolar signal is very small, also remove unipolar activation time, since it's likely not a good signal

debug_plot = 0
if debug_plot == 1:
    # plot activation time map
    xyz = voxel3mm
    lat = activation_uni.astype(float)
    lat[lat==0] = np.nan # set zero activation time to nan so they will be grey in the plot

    data = lat 
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_threshold = data_min-0.1
    converted_color = common.convert_value_to_red_blue(data, data_min, data_max, data_threshold)

    cube_half_size = 1.5  # mm, half-side of each voxel cube
    n_voxels = xyz.shape[0]
    nan_mask = np.isnan(lat)
    corner_offsets = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]
    ], dtype=float) * cube_half_size
    face_template = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,3,7],[0,7,4],[1,2,6],[1,6,5],
    ])

    def make_mesh(mask, colors, opacity):
        sub_xyz = xyz[mask]
        sub_n = sub_xyz.shape[0]
        verts = (sub_xyz[:, np.newaxis, :] + corner_offsets[np.newaxis, :, :]).reshape(-1, 3)
        base = (np.arange(sub_n) * 8)[:, np.newaxis, np.newaxis]
        fcs = (base + face_template[np.newaxis]).reshape(-1, 3)
        vcols = ['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in np.repeat(colors, 8, axis=0)]
        return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=fcs[:, 0], j=fcs[:, 1], k=fcs[:, 2], vertexcolor=vcols, opacity=opacity, flatshading=True)

    traces = []
    if np.any(~nan_mask):
        traces.append(make_mesh(~nan_mask, converted_color[~nan_mask], opacity=1.0))
    if np.any(nan_mask):
        grey = np.full((nan_mask.sum(), 3), 0.5)
        traces.append(make_mesh(nan_mask, grey, opacity=0.05))

    fig = go.Figure(data=traces)
    no_axis = dict(showgrid=False, zeroline=False, showticklabels=False, showaxeslabels=False, visible=False)
    fig.update_layout(scene=dict(aspectmode='data', xaxis=no_axis, yaxis=no_axis, zaxis=no_axis))
    fig.show()

#%% 
# save data
clinical_data = {}
for key, value in geometry_data.items():
    clinical_data[key] = value

clinical_data['electrode_positions'] = electrode_positions
clinical_data['clinical_electrogram_unipolar_original'] = clinical_electrogram_unipolar_original
clinical_data['clinical_electrogram_bipolar_original'] = clinical_electrogram_bipolar_original
clinical_data['clinical_electrogram_unipolar_refined'] = clinical_electrogram_unipolar_refined
clinical_data['clinical_electrogram_bipolar_refined'] = clinical_electrogram_bipolar_refined
clinical_data['clinical_electrogram_reference'] = clinical_electrogram_reference
clinical_data['clinical_electrogram_woi_start'] = t_start
clinical_data['clinical_electrogram_woi_end'] = t_end
clinical_data['clinical_activation_uni'] = activation_uni
clinical_data['clinical_activation_bi'] = activation_bi

file_path = directory['data_geometry'] / f'{name_prefix}_clinical_data.npz'
np.savez(file_path, **clinical_data)

print('done')

#%%
