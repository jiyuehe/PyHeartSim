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
from numba import cuda # pip install numba
from . import unipolar_electrogram_equation_parts as eq_parts

# GPU PARALLELIZED VERSION
# --------------------------------------------------
@cuda.jit
def compute_egm_kernel_batched(n_voxel, n_electrode, n_time, D11, D12, D13, D21, D22, D23, 
                                D31, D32, D33, c_voxel, l, l_x, l_y, l_z, 
                                dvdx, dvdy, dvdz, egm_out):
    """
    Optimized kernel that processes multiple time steps and uses 2D grid
    Grid: (voxels, electrodes)
    """
    voxel_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    electrode_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if voxel_idx < n_voxel and electrode_idx < n_electrode:
        # Precompute common terms
        l_val = l[voxel_idx, electrode_idx]
        l_inv_cubed = 1.0 / (l_val * l_val * l_val)
        c_val = c_voxel[voxel_idx]
        
        # Precompute diffusion tensor components
        D11_val = D11[voxel_idx]
        D12_val = D12[voxel_idx]
        D13_val = D13[voxel_idx]
        D21_val = D21[voxel_idx]
        D22_val = D22[voxel_idx]
        D23_val = D23[voxel_idx]
        D31_val = D31[voxel_idx]
        D32_val = D32[voxel_idx]
        D33_val = D33[voxel_idx]
        
        # Precompute distance components
        lx_val = l_x[voxel_idx, electrode_idx]
        ly_val = l_y[voxel_idx, electrode_idx]
        lz_val = l_z[voxel_idx, electrode_idx]
        
        # Process all time steps for this voxel-electrode pair
        for t in range(n_time):
            # Get gradients for this voxel and time step
            dvdx_val = dvdx[t, voxel_idx]
            dvdy_val = dvdy[t, voxel_idx]
            dvdz_val = dvdz[t, voxel_idx]
            
            # Compute contribution
            term_x = (D11_val * dvdx_val + D12_val * dvdy_val + D13_val * dvdz_val) * lx_val
            term_y = (D21_val * dvdx_val + D22_val * dvdy_val + D23_val * dvdz_val) * ly_val
            term_z = (D31_val * dvdx_val + D32_val * dvdy_val + D33_val * dvdz_val) * lz_val
            
            egm_contribution = c_val * l_inv_cubed * (term_x + term_y + term_z)
            
            # Atomic add to output
            cuda.atomic.add(egm_out, (t, electrode_idx), egm_contribution)

def compute(electrode_xyz, voxel, D0, c_voxel, action_potential, Delta, neighbor_id_2d, batch_size=50):
    """
    Optimized GPU execution with batched processing and reduced memory transfers
    
    Args:
        batch_size: Number of time steps to process together (default: 50)
    """
    n_voxel = voxel.shape[0]
    n_electrode = electrode_xyz.shape[0]
    T = action_potential.shape[0]
    
    # Extract diffusion tensor components (GPU uses float32)
    D11, D12, D13, D21, D22, D23, D31, D32, D33 = eq_parts.extract_diffusion_tensor_components(
        D0, n_voxel, dtype=np.float32
    )
    
    # Compute electrode distances (GPU uses float32)
    l, l_x, l_y, l_z = eq_parts.compute_electrode_distances(electrode_xyz, voxel, dtype=np.float32)
    
    # Compute all gradients at once (more efficient)
    dvdx, dvdy, dvdz = eq_parts.compute_gradients(action_potential, neighbor_id_2d, Delta)
    dvdx = dvdx.astype(np.float32)
    dvdy = dvdy.astype(np.float32)
    dvdz = dvdz.astype(np.float32)
    
    # Transfer constant data to GPU once (no broadcasting needed!)
    d_D11 = cuda.to_device(D11)
    d_D12 = cuda.to_device(D12)
    d_D13 = cuda.to_device(D13)
    d_D21 = cuda.to_device(D21)
    d_D22 = cuda.to_device(D22)
    d_D23 = cuda.to_device(D23)
    d_D31 = cuda.to_device(D31)
    d_D32 = cuda.to_device(D32)
    d_D33 = cuda.to_device(D33)
    d_c_voxel = cuda.to_device(c_voxel.astype(np.float32))
    d_l = cuda.to_device(l)
    d_l_x = cuda.to_device(l_x)
    d_l_y = cuda.to_device(l_y)
    d_l_z = cuda.to_device(l_z)
    
    # Allocate output array
    electrogram_unipolar = np.zeros((T, n_electrode), dtype=np.float32)
    
    # Process in batches to reduce memory transfers
    num_batches = (T + batch_size - 1) // batch_size
    
    # Configure 2D grid for better parallelization
    threads_per_block = (16, 16)  # 256 threads total
    blocks_x = (n_voxel + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (n_electrode + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)
    
    for batch_idx in range(num_batches):
        t_start = batch_idx * batch_size
        t_end = min(t_start + batch_size, T)
        n_time_batch = t_end - t_start
        
        if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
            print(f'compute electrogram {t_end / T * 100:.0f}%')
        
        # Transfer gradient batch to GPU (make contiguous for transfer)
        d_dvdx = cuda.to_device(np.ascontiguousarray(dvdx[t_start:t_end, :]))
        d_dvdy = cuda.to_device(np.ascontiguousarray(dvdy[t_start:t_end, :]))
        d_dvdz = cuda.to_device(np.ascontiguousarray(dvdz[t_start:t_end, :]))
        d_egm_out = cuda.to_device(np.zeros((n_time_batch, n_electrode), dtype=np.float32))
        
        # Launch kernel
        compute_egm_kernel_batched[blocks_per_grid, threads_per_block](
            n_voxel, n_electrode, n_time_batch,
            d_D11, d_D12, d_D13, d_D21, d_D22, d_D23, d_D31, d_D32, d_D33,
            d_c_voxel, d_l, d_l_x, d_l_y, d_l_z,
            d_dvdx, d_dvdy, d_dvdz, d_egm_out
        )
        
        # Copy batch result back
        electrogram_unipolar[t_start:t_end, :] = d_egm_out.copy_to_host()
    
    # Normalize electrogram magnitude
    electrogram_unipolar = eq_parts.normalize_electrogram_magnitude(electrogram_unipolar)
    
    return electrogram_unipolar
