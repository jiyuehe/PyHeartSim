#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import sys
sys.path.insert(0, str(script_dir.parent))  # add PyHeartSim directory to path
import common
from toolbox import codes as toolbox_codes
import numpy as np # pip install numpy

from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt # pip install matplotlib
import matplotlib.animation as animation

#%%
def execute(input_arguments):
    save_movie_flag = input_arguments['save_movie_flag']
    starting_time = input_arguments['starting_time']
    ending_time = input_arguments['ending_time']
    movie_save_dir = input_arguments['movie_save_dir']
    simulation_results = input_arguments['simulation_results']
    geometry_flag = simulation_results['geometry_flag']

    # load data
    geometry_data = input_arguments['geometry_data']

    if geometry_flag == 2:
        voxel_for_each_vertex = geometry_data['voxel_for_each_vertex']

        node = geometry_data['voxel'][voxel_for_each_vertex,:]
    elif geometry_flag in [0, 1, 3, 4]:
        voxel_for_each_vertex_3mm = geometry_data['voxel_for_each_vertex_3mm']

        node = geometry_data['voxel'][voxel_for_each_vertex_3mm,:]

    # simulation data
    action_potential = simulation_results['action_potential']
    t = simulation_results['physical_time']

    if ending_time == []:
        ending_time = action_potential.shape[0]

    start_id = np.argmin(np.abs(t - starting_time)) # find index of closest value
    end_id = np.argmin(np.abs(t - ending_time)) # find index of closest value
    t = t[start_id:end_id]
    action_potential = action_potential[start_id:end_id, :]

    # activation phase movie using matplotlib, with option to save as gif
    if geometry_flag == 2:
        movie_data = action_potential[:, voxel_for_each_vertex] # display on vertices
    elif geometry_flag in [0, 1, 3, 4]:
        movie_data = action_potential[:, voxel_for_each_vertex_3mm] # display on 3mm vertices

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
        color = common.convert_data_to_color.purple_yellow(data, data_min, data_max, data_threshold)
        map_color[n] = color

    fig = plt.figure(figsize=(10, 8))
    
    if geometry_flag != 0: # 3D
        ax = plt.axes(projection='3d')
        plot_handle = ax.scatter(node[:, 0], node[:, 1], node[:, 2], c=map_color[0], edgecolor='none', linewidth=0, s=140, marker='s')
        plt.axis('off')
        ax.view_init(elev=70, azim=-70)
        common.set_axes_equal.execute(ax)
    elif geometry_flag == 0: # 2D sheet
        nx = int(np.max(node[:,0]) - np.min(node[:,0])) + 1
        ny = int(np.max(node[:,1]) - np.min(node[:,1])) + 1
        color_image = map_color[0].reshape((nx, ny, 3))  # shape (30, 20, 3)
        color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow

        ax = plt.axes()
        plot_handle = ax.imshow(color_image, origin='lower', interpolation='nearest')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pause_interval = 0.001
    view_matrices = {} # dictionary to store projection matrices for each frame
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'playing movie {(n+1)/n_time*100:.0f}%')

        if geometry_flag == 0: # 2D sheet
            color_image = map_color[n].reshape((nx, ny, 3))  # shape (30, 20, 3)
            color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow
            plot_handle.set_data(color_image)
        elif geometry_flag != 0: # 3D
            plot_handle.set_color(map_color[n])
        
        ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

        # capture current view angles
        if geometry_flag != 0: # 3D
            view_matrices[n] = ax.get_proj().copy() # copy to avoid overwriting

        plt.pause(pause_interval)

    # save simulation movie
    if save_movie_flag == 1:
        def animate(n):
            if ((n+1) % (n_time//10)) == 0:
                print(f'saving movie {(n+1)/n_time*100:.0f}%')

            if geometry_flag == 0: # 2D sheet
                color_image = map_color[n].reshape((nx, ny, 3))  # shape (30, 20, 3)
                color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow
                plot_handle.set_data(color_image)
            elif geometry_flag != 0: # 3D
                plot_handle.set_color(map_color[n])

            ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

            # restore view angle
            if geometry_flag != 0: # 3D
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
