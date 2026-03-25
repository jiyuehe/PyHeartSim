import numpy as np # pip install numpy
import toolbox.codes
import matplotlib.pyplot as plt # pip install matplotlib
import common

def execute(node, signal, data_flag, geometry_flag, plot_lat_map_flag, fig_name):
    # compute local activation time map
    dvdt = np.diff(signal, axis=0)
    data_flag = 1
    if data_flag == 0: # action potential
        max_dvdt = dvdt  # positive derivative
    elif data_flag == 1: # electrogram
        max_dvdt = -dvdt  # negative derivative

    max_dvdt_indices = np.argmax(max_dvdt, axis=0)  # shape: (nodes,)
    lat = max_dvdt_indices - np.min(max_dvdt_indices) # normalize to start from 0

    if plot_lat_map_flag == 1:
        # convert local activation time into color
        data = lat
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_threshold = data_min-0.1 # a little small than data_min, so that places with value of data_min will have color
        color = common.convert_data_to_color.execute(data, data_min, data_max, data_threshold)

        if geometry_flag == 0: # 2D sheet
            nx = int(np.max(node[:,0]) - np.min(node[:,0])) + 1
            ny = int(np.max(node[:,1]) - np.min(node[:,1])) + 1
            color_image = color.reshape((nx, ny, 3))  # shape (30, 20, 3)
            color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.imshow(color_image, origin='lower', interpolation='nearest')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.tight_layout()
        elif geometry_flag in [1, 2]: # 3D slab or patient atrium          
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(node[:, 0], node[:, 1], node[:, 2], c=color, edgecolor='none', linewidth=0, s=60, marker='s')
            plt.axis('off')
            ax.view_init(elev=70, azim=-70)
            common.set_axes_equal.execute(ax)
            plt.tight_layout()

        plt.savefig(fig_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    return lat
