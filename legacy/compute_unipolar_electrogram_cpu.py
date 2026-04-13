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

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange # pip install numba
from . import unipolar_electrogram_equation_parts as eq_parts

# CPU paralleled computation
# --------------------------------------------------
@njit(parallel=True)
def compute_voxel(n_voxel, n_electrode, D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b, c_voxel, l, l_x, l_y, l_z, dvdx_b, dvdy_b, dvdz_b): 
    egm_part = np.zeros((n_voxel, n_electrode))
    for n in prange(n_voxel):
        egm_part[n,:] = c_voxel[n] / (l[n,:]**3) * (
            (D11_b[n,:] * dvdx_b[n,:] + D12_b[n,:] * dvdy_b[n,:] + D13_b[n,:] * dvdz_b[n,:]) * l_x[n,:] +
            (D21_b[n,:] * dvdx_b[n,:] + D22_b[n,:] * dvdy_b[n,:] + D23_b[n,:] * dvdz_b[n,:]) * l_y[n,:] +
            (D31_b[n,:] * dvdx_b[n,:] + D32_b[n,:] * dvdy_b[n,:] + D33_b[n,:] * dvdz_b[n,:]) * l_z[n,:]
        )

    return egm_part
        
def execute(electrode_xyz, voxel, D0, c_voxel, action_potential, Delta, neighbor_id_2d):
    n_voxel = voxel.shape[0]
    n_electrode = electrode_xyz.shape[0]
    
    # Extract diffusion tensor components
    D11, D12, D13, D21, D22, D23, D31, D32, D33 = eq_parts.extract_diffusion_tensor_components(D0, n_voxel)
    
    # Compute electrode distances
    l, l_x, l_y, l_z = eq_parts.compute_electrode_distances(electrode_xyz, voxel)
    
    # Compute gradients
    dvdx, dvdy, dvdz = eq_parts.compute_gradients(action_potential, neighbor_id_2d, Delta)
    
    # Broadcast diffusion tensors
    D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b = eq_parts.broadcast_diffusion_tensors(
        D11, D12, D13, D21, D22, D23, D31, D32, D33, n_electrode
    )

    T = action_potential.shape[1]
    electrogram_unipolar = np.zeros((n_electrode, T))
    for t_id in range(T):
        if (t_id + 1) % (T // 5) == 0:
            print(f'compute electrogram {(t_id + 1) / T * 100:.0f}%')
        
        dvdx_b = np.tile(dvdx[:, t_id].reshape(-1, 1), (1, n_electrode))
        dvdy_b = np.tile(dvdy[:, t_id].reshape(-1, 1), (1, n_electrode))
        dvdz_b = np.tile(dvdz[:, t_id].reshape(-1, 1), (1, n_electrode))
        
        egm_part = compute_voxel(n_voxel, n_electrode, D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b, c_voxel, l, l_x, l_y, l_z, dvdx_b, dvdy_b, dvdz_b)

        electrogram_unipolar[:, t_id] = np.sum(egm_part, axis=0)
    
    # Normalize electrogram magnitude
    electrogram_unipolar = eq_parts.normalize_electrogram_magnitude(electrogram_unipolar)
    
    debug_plot = 0
    if debug_plot == 1:
        plt.figure()
        # e_id = 0
        # plt.plot(electrogram_unipolar[e_id,:], 'b')
        plt.plot(electrogram_unipolar.T)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (scaled)')
        plt.title('Unipolar electrogram')
        plt.show()
    
    return electrogram_unipolar
