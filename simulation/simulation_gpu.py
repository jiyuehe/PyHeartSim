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
from numba import cuda
from simulation.pacing import assign_pacing_parameters, apply_pacing

# CuPy for GPU sparse operations
try: # need this try-except for MacOS compatibility
    import cupy as cp # pip install cupy-cuda12x (NOTE: replace '12x' with your CUDA version)
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_linalg
except ImportError: 
    print("GPU is not available.")

@cuda.jit(device=True)
def compute_reaction_only_gpu(heart_model_flag, u, h, P_2d, J_stim, n):
    if heart_model_flag == 0: # Mitchell-Schaeffer model
        du_dt = (h[n] * (u[n]**2) * (1 - u[n])) / P_2d[n, 17] - \
                u[n] / P_2d[n, 18] + J_stim[n]
        
        if u[n] < P_2d[n, 19]:
            dh_dt = (1 - h[n]) / P_2d[n, 15]
        else:
            dh_dt = -h[n] / P_2d[n, 16]
            
    else: # heart_model_flag == 1: Aliev-Panfilov model
        du_dt = -P_2d[n, 15]*u[n]*(u[n] - P_2d[n, 16])*(u[n] - 1) - u[n]*h[n] + J_stim[n]
        dh_dt = (P_2d[n, 17] + P_2d[n, 18]*h[n]/(u[n] + P_2d[n, 19])) * \
                (-h[n] - P_2d[n, 15]*u[n]*(u[n] - P_2d[n, 16] - 1))
    
    return du_dt, dh_dt

@cuda.jit
def reaction_k1_kernel(heart_model_flag, u_current, h_current, k1_u, k1_h, P_2d, J_stim):
    n = cuda.grid(1)
    if n < u_current.shape[0]:
        k1_u[n], k1_h[n] = compute_reaction_only_gpu(heart_model_flag, u_current, h_current, P_2d, J_stim, n)

@cuda.jit
def reaction_temp_kernel(u_current, h_current, u_temp, h_temp, k_u, k_h, dt, factor):
    n = cuda.grid(1)
    if n < u_current.shape[0]:
        u_temp[n] = u_current[n] + factor * dt * k_u[n]
        h_temp[n] = h_current[n] + factor * dt * k_h[n]

@cuda.jit
def reaction_k_kernel(heart_model_flag, u_temp, h_temp, k_u, k_h, P_2d, J_stim):
    n = cuda.grid(1)
    if n < u_temp.shape[0]:
        k_u[n], k_h[n] = compute_reaction_only_gpu(heart_model_flag, u_temp, h_temp, P_2d, J_stim, n)

@cuda.jit
def reaction_final_kernel(u_current, h_current, u_star, h_next, k1_u, k1_h, k2_u, k2_h, k3_u, k3_h, k4_u, k4_h, dt):
    n = cuda.grid(1)
    if n < u_current.shape[0]:
        u_star[n] = u_current[n] + (dt / 6.0) * (k1_u[n] + 2.0*k2_u[n] + 2.0*k3_u[n] + k4_u[n])
        h_next[n] = h_current[n] + (dt / 6.0) * (k1_h[n] + 2.0*k2_h[n] + 2.0*k3_h[n] + k4_h[n])

@cuda.jit
def rk4_reaction_kernel(heart_model_flag, u_current, h_current, u_star, h_next, P_2d, J_stim, dt):
    n = cuda.grid(1)
    if n < u_current.shape[0]:
        u_n = u_current[n]
        h_n = h_current[n]
        
        # K1
        k1_u, k1_h = compute_reaction_only_gpu(heart_model_flag, u_current, h_current, P_2d, J_stim, n)
        
        # K2: use temp values stored in registers
        u_temp = u_n + 0.5 * dt * k1_u
        h_temp = h_n + 0.5 * dt * k1_h
        
        # For k2, we need to compute reaction with temp values
        # Create local computation using the temp values
        if heart_model_flag == 0:  # Mitchell-Schaeffer
            k2_u = (h_temp * (u_temp**2) * (1 - u_temp)) / P_2d[n, 17] - u_temp / P_2d[n, 18] + J_stim[n]
            if u_temp < P_2d[n, 19]:
                k2_h = (1 - h_temp) / P_2d[n, 15]
            else:
                k2_h = -h_temp / P_2d[n, 16]
        else:  # Aliev-Panfilov
            k2_u = -P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16])*(u_temp - 1) - u_temp*h_temp + J_stim[n]
            k2_h = (P_2d[n, 17] + P_2d[n, 18]*h_temp/(u_temp + P_2d[n, 19])) * \
                   (-h_temp - P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16] - 1))
        
        # K3
        u_temp = u_n + 0.5 * dt * k2_u
        h_temp = h_n + 0.5 * dt * k2_h
        
        if heart_model_flag == 0:
            k3_u = (h_temp * (u_temp**2) * (1 - u_temp)) / P_2d[n, 17] - u_temp / P_2d[n, 18] + J_stim[n]
            if u_temp < P_2d[n, 19]:
                k3_h = (1 - h_temp) / P_2d[n, 15]
            else:
                k3_h = -h_temp / P_2d[n, 16]
        else:
            k3_u = -P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16])*(u_temp - 1) - u_temp*h_temp + J_stim[n]
            k3_h = (P_2d[n, 17] + P_2d[n, 18]*h_temp/(u_temp + P_2d[n, 19])) * \
                   (-h_temp - P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16] - 1))
        
        # K4
        u_temp = u_n + dt * k3_u
        h_temp = h_n + dt * k3_h
        
        if heart_model_flag == 0:
            k4_u = (h_temp * (u_temp**2) * (1 - u_temp)) / P_2d[n, 17] - u_temp / P_2d[n, 18] + J_stim[n]
            if u_temp < P_2d[n, 19]:
                k4_h = (1 - h_temp) / P_2d[n, 15]
            else:
                k4_h = -h_temp / P_2d[n, 16]
        else:
            k4_u = -P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16])*(u_temp - 1) - u_temp*h_temp + J_stim[n]
            k4_h = (P_2d[n, 17] + P_2d[n, 18]*h_temp/(u_temp + P_2d[n, 19])) * \
                   (-h_temp - P_2d[n, 15]*u_temp*(u_temp - P_2d[n, 16] - 1))
        
        # Final combination
        u_star[n] = u_n + (dt / 6.0) * (k1_u + 2.0*k2_u + 2.0*k3_u + k4_u)
        h_next[n] = h_n + (dt / 6.0) * (k1_h + 2.0*k2_h + 2.0*k3_h + k4_h)

def reaction_step_gpu_rk4(heart_model_flag, d_u_current, d_h_current, d_u_star, d_h_next, 
                                 d_P_2d, d_J_stim, dt, threads_per_block, blocks_per_grid):
    rk4_reaction_kernel[blocks_per_grid, threads_per_block](
        heart_model_flag, d_u_current, d_h_current, d_u_star, d_h_next, d_P_2d, d_J_stim, dt
    )

def build_diffusion_matrix_gpu(P_2d, neighbor_id_2d_2, Delta):
    n_voxel = P_2d.shape[0]
    
    # Build on CPU first (this is fast enough)
    row_indices = []
    col_indices = []
    data_values = []
    
    for n in range(n_voxel):
        D_coeff = P_2d[n, 20] / (4 * Delta**2)
        
        # Diagonal entry
        diagonal_sum = -(P_2d[n, 0] + P_2d[n, 1] + P_2d[n, 2] + 
                        P_2d[n, 3] + P_2d[n, 4] + P_2d[n, 5])
        row_indices.append(n)
        col_indices.append(n)
        data_values.append(D_coeff * diagonal_sum)
        
        # Direct neighbors
        neighbor_coeffs = [
            (neighbor_id_2d_2[n, 0], P_2d[n, 0]),
            (neighbor_id_2d_2[n, 1], P_2d[n, 1]),
            (neighbor_id_2d_2[n, 2], P_2d[n, 2]),
            (neighbor_id_2d_2[n, 3], P_2d[n, 3]),
            (neighbor_id_2d_2[n, 4], P_2d[n, 4]),
            (neighbor_id_2d_2[n, 5], P_2d[n, 5]),
        ]
        
        for neighbor_id, coeff in neighbor_coeffs:
            if neighbor_id >= 0 and neighbor_id < n_voxel:
                row_indices.append(n)
                col_indices.append(neighbor_id)
                data_values.append(D_coeff * coeff)
        
        # Cross-derivative terms
        if neighbor_id_2d_2[n, 0] >= 0 and neighbor_id_2d_2[n, 0] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 0])
            data_values.append(D_coeff * P_2d[n, 6])
        if neighbor_id_2d_2[n, 1] >= 0 and neighbor_id_2d_2[n, 1] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 1])
            data_values.append(-D_coeff * P_2d[n, 6])
        
        if neighbor_id_2d_2[n, 2] >= 0 and neighbor_id_2d_2[n, 2] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 2])
            data_values.append(D_coeff * P_2d[n, 7])
        if neighbor_id_2d_2[n, 3] >= 0 and neighbor_id_2d_2[n, 3] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 3])
            data_values.append(-D_coeff * P_2d[n, 7])
        
        if neighbor_id_2d_2[n, 4] >= 0 and neighbor_id_2d_2[n, 4] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 4])
            data_values.append(D_coeff * P_2d[n, 8])
        if neighbor_id_2d_2[n, 5] >= 0 and neighbor_id_2d_2[n, 5] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 5])
            data_values.append(-D_coeff * P_2d[n, 8])
        
        cross_terms = [
            (neighbor_id_2d_2[n, 6], neighbor_id_2d_2[n, 8], P_2d[n, 9]),
            (neighbor_id_2d_2[n, 9], neighbor_id_2d_2[n, 7], P_2d[n, 10]),
            (neighbor_id_2d_2[n, 14], neighbor_id_2d_2[n, 16], P_2d[n, 11]),
            (neighbor_id_2d_2[n, 17], neighbor_id_2d_2[n, 15], P_2d[n, 12]),
            (neighbor_id_2d_2[n, 10], neighbor_id_2d_2[n, 12], P_2d[n, 13]),
            (neighbor_id_2d_2[n, 13], neighbor_id_2d_2[n, 11], P_2d[n, 14]),
        ]
        
        for nb1, nb2, coeff in cross_terms:
            if nb1 >= 0 and nb1 < n_voxel:
                row_indices.append(n)
                col_indices.append(nb1)
                data_values.append(D_coeff * coeff)
            if nb2 >= 0 and nb2 < n_voxel:
                row_indices.append(n)
                col_indices.append(nb2)
                data_values.append(-D_coeff * coeff)
    
    # Create sparse matrix and transfer to GPU
    row_indices = cp.array(row_indices, dtype=cp.int32)
    col_indices = cp.array(col_indices, dtype=cp.int32)
    data_values = cp.array(data_values, dtype=cp.float32)
    
    L_gpu = cp_sparse.coo_matrix((data_values, (row_indices, col_indices)), 
                                  shape=(n_voxel, n_voxel))
    return L_gpu.tocsr()

def crank_nicolson_diffusion_step_gpu(u_star_gpu, L_matrix_gpu, dt, method, A_gpu_cached, tol=1e-6):
    """
    Solve implicit diffusion step using Crank-Nicolson on GPU
    
    Solves: (I - dt/2 * L) * u_next = u_star + dt/2 * L * u_star
    
    Args:
        u_star_gpu: CuPy array - voltage after reaction step
        L_matrix_gpu: CuPy sparse CSR matrix - diffusion operator
        dt: time step
        method: 'cg' (Conjugate Gradient) or 'gmres'
        A_gpu_cached: Pre-computed (I - dt/2 * L) matrix
        tol: tolerance for iterative solvers
    
    Returns:
        u_next: CuPy array - voltage after diffusion step
    """
    # Compute RHS: b = u_star + dt/2 * L * u_star
    b_gpu = u_star_gpu + (dt / 2.0) * (L_matrix_gpu @ u_star_gpu)
    
    # Solve linear system on GPU using available CuPy solvers
    if method == 'cg':
        u_next, info = cp_linalg.cg(A_gpu_cached, b_gpu, x0=u_star_gpu, atol=tol, maxiter=10000)
        if info != 0:
            print(f"Warning: GPU CG did not converge (info={info})")
    elif method == 'gmres':
        u_next, info = cp_linalg.gmres(A_gpu_cached, b_gpu, x0=u_star_gpu, atol=tol, maxiter=10000, restart=50)
        if info != 0:
            print(f"Warning: GPU GMRES did not converge (info={info})")
    else:
        raise ValueError(f"Unknown solver method: {method}. Use 'cg' or 'gmres'")
    
    return u_next

def compute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters):
    ##############################
    # for macro re-entry
    node_flag = arrhythmia_parameters['node_flag']
    block_voxel_id = np.where(node_flag == 3)[0]
    ##############################

    # geometry data
    neighbor_id_2d = geometry_data['neighbor_id_2d']
    Delta = geometry_data['Delta']
    
    # simulation parameters
    t_final = float(simulation_parameters['t_final'])
    dt = simulation_parameters['dt']
    arrhythmia_flag = simulation_parameters['arrhythmia_flag']
    
    # pacing parameters
    J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t = assign_pacing_parameters(arrhythmia_parameters, arrhythmia_flag, n_voxel, neighbor_id_2d, simulation_parameters)
    
    # set initial value at rest
    if simulation_parameters['heart_model_flag'] == 0:
        u_current = np.zeros(n_voxel, dtype=np.float32)
        h_current = np.ones(n_voxel, dtype=np.float32)
    elif simulation_parameters['heart_model_flag'] == 1:
        u_current = np.zeros(n_voxel, dtype=np.float32)
        h_current = np.zeros(n_voxel, dtype=np.float32)
    
    # calculate the number of samples needed for 1 kHz sampling
    if simulation_parameters['heart_model_flag'] == 0:
        n_samples = int(np.round(t_final))
    elif simulation_parameters['heart_model_flag'] == 1:
        n_samples = int(np.round(t_final * simulation_parameters['time_scale']))
    
    sim_u_voxel = np.zeros((n_samples, n_voxel), dtype=np.float32)
    sim_h_voxel = np.zeros((n_samples, n_voxel), dtype=np.float32)
    physical_time = np.zeros(n_samples, dtype=np.float32)
    
    neighbor_id_2d_2 = neighbor_id_2d.copy()
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0
    
    # GPU configuration
    threads_per_block = 256
    blocks_per_grid = (n_voxel + threads_per_block - 1) // threads_per_block
    
    # Convert to float32 and ensure contiguous arrays
    P_2d = np.ascontiguousarray(P_2d.astype(np.float32))
    neighbor_id_2d_2 = np.ascontiguousarray(neighbor_id_2d_2.astype(np.int32))
    J_stim = np.ascontiguousarray(J_stim.astype(np.float32))
    dt_float = np.float32(dt)
    Delta_float = np.float32(Delta)
    
    # Build diffusion matrix on GPU (only once)
    L_matrix_gpu = build_diffusion_matrix_gpu(P_2d, neighbor_id_2d_2, Delta_float)
    
    # Pre-compute the Crank-Nicolson system matrix (I - dt/2 * L) - only once!
    I_gpu = cp_sparse.identity(n_voxel, format='csr', dtype=cp.float32)
    A_gpu_cached = I_gpu - (dt_float / 2.0) * L_matrix_gpu

    # Pre-build blocked diffusion matrix (P_2d[:, 20] = 0 for block_voxel_id)
    if len(block_voxel_id) > 0:
        P_2d_blocked = P_2d.copy()
        P_2d_blocked[block_voxel_id, 20] = 0.0
        L_matrix_gpu_blocked = build_diffusion_matrix_gpu(P_2d_blocked, neighbor_id_2d_2, Delta_float)
        A_gpu_cached_blocked = I_gpu - (dt_float / 2.0) * L_matrix_gpu_blocked
    else:
        L_matrix_gpu_blocked = L_matrix_gpu
        A_gpu_cached_blocked = A_gpu_cached
    
    # Allocate GPU memory for CUDA kernels (CuPy arrays for unified memory access)
    d_u_current = cp.zeros(n_voxel, dtype=cp.float32)
    d_h_current = cp.zeros(n_voxel, dtype=cp.float32)
    d_u_star = cp.zeros(n_voxel, dtype=cp.float32)
    d_h_next = cp.zeros(n_voxel, dtype=cp.float32)
    d_P_2d = cp.asarray(P_2d)
    d_J_stim = cp.zeros(n_voxel, dtype=cp.float32)
    
    # Initialize with starting values
    d_u_current[:] = cp.asarray(u_current)
    d_h_current[:] = cp.asarray(h_current)
    
    # Get CUDA device arrays from CuPy for numba kernels
    cuda_u_current = cuda.as_cuda_array(d_u_current)
    cuda_h_current = cuda.as_cuda_array(d_h_current)
    cuda_u_star = cuda.as_cuda_array(d_u_star)
    cuda_h_next = cuda.as_cuda_array(d_h_next)
    cuda_P_2d = cuda.as_cuda_array(d_P_2d)
    cuda_J_stim = cuda.as_cuda_array(d_J_stim)
    
    id_save = 0
    total_model_time_steps = int(np.round(t_final / dt))
    number_of_steps_per_ms = int(1 / (dt * simulation_parameters['time_scale']))
    
    for model_time_step in range(total_model_time_steps):
        if ((model_time_step+1) % (total_model_time_steps//5)) == 0:
            print(f'simulating {(model_time_step+1)/total_model_time_steps*100:.1f}%')
        
        model_time = model_time_step * dt

        # ##############################
        # for macro re-entry
        if model_time >= 0 and model_time < 350 * dt:
            active_L = L_matrix_gpu_blocked
            active_A = A_gpu_cached_blocked
        else:
            active_L = L_matrix_gpu
            active_A = A_gpu_cached
        # ##############################
        
        # apply pacing - this is CPU-side since it has complex conditionals
        J_stim.fill(0.0)
        J_stim = apply_pacing(arrhythmia_parameters, simulation_parameters, arrhythmia_flag, model_time, 
                             J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, 
                             pacing_duration, s2_t,
                             sim_u_voxel, sim_h_voxel, neighbor_id_2d)
        
        # Transfer J_stim to GPU
        d_J_stim[:] = cp.asarray(J_stim)
        
        # Operator splitting:
        # Step 1: Reaction step (GPU RK4 - fused kernel, single launch)
        reaction_step_gpu_rk4(
            simulation_parameters['heart_model_flag'],
            cuda_u_current, cuda_h_current, cuda_u_star, cuda_h_next,
            cuda_P_2d, cuda_J_stim, dt_float, threads_per_block, blocks_per_grid
        )
        
        # Step 2: Diffusion step (GPU Crank-Nicolson) - d_u_star already on GPU
        u_next = crank_nicolson_diffusion_step_gpu(d_u_star, active_L, dt_float, method='gmres', A_gpu_cached=active_A, tol=1e-5)
        
        # Update for next iteration (stay on GPU)
        d_u_current[:] = u_next
        d_h_current[:] = d_h_next
        
        # save value at 1 kHz - only copy to CPU when saving
        if number_of_steps_per_ms > 0:
            if (model_time_step % number_of_steps_per_ms) == 0 and id_save < n_samples:
                sim_u_voxel[id_save, :] = cp.asnumpy(d_u_current)
                sim_h_voxel[id_save, :] = cp.asnumpy(d_h_current)
                physical_time[id_save] = model_time * simulation_parameters['time_scale']
                id_save = id_save + 1
        else:
            # If dt is too large, save every step
            if id_save < n_samples:
                sim_u_voxel[id_save, :] = cp.asnumpy(d_u_current)
                sim_h_voxel[id_save, :] = cp.asnumpy(d_h_current)
                physical_time[id_save] = model_time * simulation_parameters['time_scale']
                id_save = id_save + 1

    return sim_u_voxel[:id_save, :], sim_h_voxel[:id_save, :], physical_time[:id_save]
