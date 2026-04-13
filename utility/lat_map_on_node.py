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

import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from matplotlib.tri import Triangulation
from . import common

def execute(geometry, signal, data_flag, geometry_flag, plot_lat_map_flag, fig_name):
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
            node = geometry['node']
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
            node = geometry['node']
            # face = geometry['face']
            # vertex = geometry['vertex']

            # plot_type = 'scatter' # 'scatter' or 'trisurf'
            # if plot_type == 'scatter':
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(node[:, 0], node[:, 1], node[:, 2], c=color, edgecolor='none', linewidth=0, s=60, marker='s')
            plt.axis('off')
            ax.view_init(elev=70, azim=-70)
            common.set_axes_equal.execute(ax)
            plt.tight_layout()
            # elif plot_type == 'trisurf':                
            #     triang = Triangulation(vertex[:, 0], vertex[:, 1], triangles=face)
            #     plt.figure()
            #     ax = plt.axes(projection='3d')
            #     face_color = color[face].mean(axis=1) # Convert per-vertex RGB colors to per-triangle colors for trisurf.
            #     surf = ax.plot_trisurf(triang, vertex[:, 2], edgecolor='gray', linewidth=0, antialiased=False, shade=False)
            #     surf.set_facecolor(face_color)
            #     plt.axis('off')
            #     ax.view_init(elev=70, azim=-70)
            #     common.set_axes_equal.execute(ax)
            #     plt.tight_layout()

        plt.savefig(fig_name, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    return lat
