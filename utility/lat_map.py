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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator # pip install scipy
import common

def compute_electrode_lat(electrode_signal):
    dvdt = np.diff(electrode_signal, axis=0)
    max_dvdt = -dvdt  # negative derivative

    max_dvdt_indices = np.argmax(max_dvdt, axis=0)  # shape: (electrodes,)
    lat = max_dvdt_indices - np.min(max_dvdt_indices) # normalize to start from 0

    return lat

def interpolate_lat(voxel, electrode_voxel, lat_electrode):
    # Remove constant dimensions to avoid QhullError when points are coplanar
    col_range = np.ptp(electrode_voxel, axis=0)
    active_dims = col_range > 0
    electrode_voxel_reduced = electrode_voxel[:, active_dims]
    voxel_reduced = voxel[:, active_dims]

    # Linear interpolation using Delaunay triangulation of electrode locations
    linear_interp = LinearNDInterpolator(electrode_voxel_reduced, lat_electrode)
    lat_voxel = linear_interp(voxel_reduced)

    # Fall back to nearest-neighbour for voxels outside the convex hull (NaN)
    nan_mask = np.isnan(lat_voxel)
    if np.any(nan_mask):
        nearest_interp = NearestNDInterpolator(electrode_voxel_reduced, lat_electrode)
        lat_voxel[nan_mask] = nearest_interp(voxel_reduced[nan_mask])

    return lat_voxel

def plot(voxel, lat_voxel, geometry_flag, fig_name):
    data = lat_voxel
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_threshold = data_min-0.1 # a little small than data_min, so that places with value of data_min will have color
    color = common.convert_value_to_red_blue(data, data_min, data_max, data_threshold)
    
    # Map the voxel centers to a filled grid.  ax.voxels only draws exposed
    # faces, so adjacent cubes share a boundary instead of showing a gap.
    axis_steps = [
        np.diff(np.unique(voxel[:, axis]))
        for axis in range(3)
    ]
    positive_steps = [steps[steps > 0] for steps in axis_steps if steps.size]
    spacing = min(np.min(steps) for steps in positive_steps) if positive_steps else 1.0

    origin = np.min(voxel, axis=0)
    voxel_index = np.rint((voxel - origin) / spacing).astype(int)
    grid_shape = tuple(np.max(voxel_index, axis=0) + 1)

    filled = np.zeros(grid_shape, dtype=bool)
    filled[tuple(voxel_index.T)] = True

    rgba = np.column_stack((color, np.ones(len(color))))
    facecolors = np.zeros(grid_shape + (4,), dtype=float)
    facecolors[tuple(voxel_index.T)] = rgba

    grid_x, grid_y, grid_z = np.indices(np.asarray(grid_shape) + 1, dtype=float)
    grid_x = origin[0] + (grid_x - 0.5) * spacing
    grid_y = origin[1] + (grid_y - 0.5) * spacing
    grid_z = origin[2] + (grid_z - 0.5) * spacing

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(
        grid_x, grid_y, grid_z, filled,
        facecolors=facecolors, edgecolor='none', linewidth=0,
        shade=False, antialiased=False,
    )
    plt.axis('off')

    if geometry_flag in [1, 2]: # 3D geometry and long slab
        ax.view_init(elev=70, azim=-70)
    elif geometry_flag == 0: # 2D geometry
        ax.view_init(elev=90, azim=-90)
    
    common.set_axes_equal(ax)
    plt.tight_layout()

    plt.savefig(fig_name, dpi=300)
    plt.close()
