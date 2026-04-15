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
from . import common

#%%
def execute(in_arg):
    save_movie_flag = in_arg['save_movie_flag']
    starting_time = in_arg['starting_time']
    ending_time = in_arg['ending_time']
    movie_save_dir = in_arg['movie_save_dir']
    simulation_results = in_arg['simulation_results']
    geometry_flag = simulation_results['geometry_flag']
    geometry_data = in_arg['geometry_data']

    voxel = geometry_data['voxel']
    voxel_id_of_electrode = geometry_data['voxel_id_of_electrode']
    voxel_electrode = voxel[voxel_id_of_electrode, :]

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
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'compute color map {(n+1)/n_time*100:.0f}%')
        data = movie_data[n, :]
        color = common.convert_value_to_purple_yellow(data, data_min, data_max, data_threshold)
        map_color[n] = color

    fig = plt.figure(figsize=(10, 8))
    
    ax = plt.axes(projection='3d')
    ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], c='gray', edgecolor='none', linewidth=0, s=10, marker='s')
    plot_handle = ax.scatter(voxel_electrode[:, 0], voxel_electrode[:, 1], voxel_electrode[:, 2], c=map_color[0], edgecolor='none', linewidth=0, s=10, marker='s')
    plt.axis('off')

    if geometry_flag in [1, 2]: # 3D geometry and long slab
        ax.view_init(elev=70, azim=-70)
    elif geometry_flag == 0: # 2D geometry
        ax.view_init(elev=90, azim=-90)

    common.set_axes_equal(ax)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pause_interval = 0.001
    view_matrices = {} # dictionary to store projection matrices for each frame
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'playing movie {(n+1)/n_time*100:.0f}%')

        plot_handle.set_color(map_color[n])
        
        ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

        # capture current view angles
        view_matrices[n] = ax.get_proj().copy() # copy to avoid overwriting

        plt.pause(pause_interval)

    # save simulation movie
    if save_movie_flag == 1:
        def animate(n):
            if ((n+1) % (n_time//10)) == 0:
                print(f'saving movie {(n+1)/n_time*100:.0f}%')

            plot_handle.set_color(map_color[n])

            ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

            # restore view angle
            R = view_matrices[n]
            ax.get_proj = lambda R=R: R # use default argument to capture current R

        frame_skip = 5
        anim = animation.FuncAnimation(fig, animate, frames=range(0, n_time, frame_skip), interval=1, blit=False, repeat=False)
        # the 'interval' parameter specifies the delay between frames in milliseconds

        # save
        writer = PillowWriter(fps=20)
        anim.save(movie_save_dir, writer=writer, dpi=60)

        print("movie is saved")

    print('done')
