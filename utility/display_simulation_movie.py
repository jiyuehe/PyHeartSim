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

import numpy as np # pip install numpy

from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt # pip install matplotlib
import matplotlib.animation as animation
from .. import common

def _ensure_interactive_backend():
    """Switch from Agg to an interactive backend when a display is available."""
    backend = str(plt.get_backend()).lower()
    is_noninteractive_agg = backend in ('agg', 'module://matplotlib.backends.backend_agg')
    has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

    if not has_display:
        return

    if not is_noninteractive_agg:
        plt.ion()
        return

    # Try common GUI backends in order; continue if one is unavailable.
    for candidate in ('QtAgg', 'TkAgg', 'GTK3Agg'):
        try:
            plt.switch_backend(candidate)
            plt.ion()
            print(f"Matplotlib backend switched to {plt.get_backend()} for interactive display")
            return
        except Exception:
            continue

    print(
        "Warning: interactive Matplotlib backend not available. "
        "Install Qt or Tk support to enable interactive movie playback."
    )

#%%
def execute(in_arg):
    _ensure_interactive_backend()

    save_movie_flag = in_arg['save_movie_flag']
    starting_time = in_arg['starting_time']
    ending_time = in_arg['ending_time']
    movie_save_dir = in_arg['movie_save_dir']
    simulation_results = in_arg['simulation_results']
    geometry_flag = simulation_results['geometry_flag']
    geometry_data = in_arg['geometry_data']

    voxel = geometry_data['voxel']

    if in_arg['save_action_potential_of_all_voxel_flag'] == 1:
        voxel_valid = voxel
        action_potential = simulation_results['action_potential']
    else:
        voxel_id_of_electrode = geometry_data['voxel_id_of_electrode']
        voxel_valid = voxel[voxel_id_of_electrode, :]
        action_potential = simulation_results['action_potential_electrode']

    t = simulation_results['physical_time']

    if ending_time == []:
        ending_time = action_potential.shape[0]

    start_id = np.argmin(np.abs(t - starting_time)) # find index of closest value
    end_id = np.argmin(np.abs(t - ending_time)) # find index of closest value
    t = t[start_id:end_id]
    action_potential = action_potential[start_id:end_id, :]

    # activation phase movie using matplotlib, with option to save as gif
    movie_data = action_potential
    v_gate = 0.13
    data_min = 0 
    data_max = 1 # action potential value can be large at the pacing site, that's why cap it 1 here so that the colors are good
    data_threshold = v_gate
    map_color = {}
    n_time = movie_data.shape[0]
    frame_skip = 10
    frames = range(0, n_time, frame_skip)
    n_frames = len(frames)
    for i, n in enumerate(frames):
        if (i+1) % max(1, n_frames//5) == 0:
            print(f'compute color map {(i+1)/n_frames*100:.0f}%')
        data = movie_data[n, :]
        color = common.convert_value_to_purple_yellow(data, data_min, data_max, data_threshold)
        map_color[n] = color

    fig = plt.figure(figsize=(10, 8))
    
    ax = plt.axes(projection='3d')
    if in_arg['save_action_potential_of_all_voxel_flag'] == 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], c='gray', edgecolor='none', linewidth=0, s=10, marker='s')
    plot_handle = ax.scatter(voxel_valid[:, 0], voxel_valid[:, 1], voxel_valid[:, 2], c=map_color[0], edgecolor='none', linewidth=0, s=10, marker='s')
    plt.axis('off')

    if geometry_flag in [1, 2]: # 3D geometry and long slab
        ax.view_init(elev=70, azim=-70)
    elif geometry_flag == 0: # 2D geometry
        ax.view_init(elev=90, azim=-90)

    common.set_axes_equal(ax)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pause_interval = 0.001
    view_matrices = {} # dictionary to store projection matrices for each frame
    for i, n in enumerate(frames):
        if (i+1) % max(1, n_frames//5) == 0:
            print(f'playing movie {(i+1)/n_frames*100:.0f}%')

        plot_handle.set_color(map_color[n])
        
        ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

        # capture current view angles
        view_matrices[n] = ax.get_proj().copy() # copy to avoid overwriting

        plt.pause(pause_interval)

    # save simulation movie
    if save_movie_flag == 1:
        # compute colors for save frames not already computed
        for n in frames:
            if n not in map_color:
                data = movie_data[n, :]
                map_color[n] = common.convert_value_to_purple_yellow(data, data_min, data_max, data_threshold)

        save_counter = [0]
        def animate(n):
            save_counter[0] += 1
            if save_counter[0] % max(1, n_frames//10) == 0:
                print(f'saving movie {save_counter[0]/n_frames*100:.0f}%')

            plot_handle.set_color(map_color[n])

            ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

            # restore view angle
            if n in view_matrices:
                R = view_matrices[n]
                ax.get_proj = lambda R=R: R # use default argument to capture current R

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1, blit=False, repeat=False)
        # the 'interval' parameter specifies the delay between frames in milliseconds

        # save
        writer = PillowWriter(fps=20)
        anim.save(movie_save_dir, writer=writer, dpi=60)

        print("movie is saved")

    print('done')
