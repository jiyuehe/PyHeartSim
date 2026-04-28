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
from scipy.spatial import cKDTree
from . import lat_map

def compute(simulation_results, geometry_data):
    # compute local activation time
    electrogram_unipolar = simulation_results['electrogram_unipolar']
    lat_electrode = lat_map.compute_electrode_lat(electrogram_unipolar)

    voxel_id_of_simulation_electrode = geometry_data['voxel_id_of_simulation_electrode']
    xyz = geometry_data['voxel'][voxel_id_of_simulation_electrode, :] # coordinates of electrode voxels
    lat = lat_electrode

    # compute conduction velocity for each point
    tree = cKDTree(xyz)
    n_points = xyz.shape[0]
    conduction_velocity_vectors = np.zeros((n_points, 3))
    conduction_velocity_magnitudes = np.zeros(n_points)
    neighbor_radius = 8.0 # mm

    for i in range(n_points):
        # find neighbors (including self)
        idx = tree.query_ball_point(xyz[i], neighbor_radius)
        if len(idx) < 4:
            # not enough neighbors for 3D fit
            conduction_velocity_vectors[i, :] = np.nan
            conduction_velocity_magnitudes[i] = np.nan
            continue
        
        pts = xyz[idx]
        lats = lat[idx]

        # least-squares fit: lat = a*x + b*y + c*z + d
        A = np.column_stack((pts, np.ones(len(pts))))
        coeffs, _, _, _ = np.linalg.lstsq(A, lats, rcond=None)
        grad = coeffs[:3]  # gradient of activation time

        # conduction velocity vector: v = grad_t / |grad_t|^2
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq > 1e-8:
            v_vec = grad / grad_norm_sq
            v_mag = 1.0 / np.linalg.norm(grad)
            conduction_velocity_vectors[i, :] = v_vec
            conduction_velocity_magnitudes[i] = v_mag
        else:
            conduction_velocity_vectors[i, :] = np.nan
            conduction_velocity_magnitudes[i] = np.nan

    # mean conduction velocity
    conduction_velocity_mean = np.nanmean(conduction_velocity_magnitudes)

    return conduction_velocity_vectors, conduction_velocity_magnitudes, conduction_velocity_mean
