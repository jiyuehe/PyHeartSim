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
from simulation.phase_field import build_diffusion_matrix as build_phase_diffusion_matrix
from simulation.phase_field import diffusion_substeps

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


def prepare_diffusion_system_gpu(P_2d, neighbors, geometry_data, dt):
    """Cache a mass-weighted phase-field system; both geometry arrays are required."""
    has_phi = 'phase_field' in geometry_data
    has_faces = 'phase_field_face_fraction' in geometry_data
    # Legacy check allowed both arrays to be absent:
    # if has_phi != has_faces:
    if not has_phi or not has_faces:
        raise ValueError('Phase-field geometry requires both volume and face fractions')
    # n = len(P_2d)  # Only needed by the disabled legacy fallback.
    Delta = float(geometry_data['Delta'])
    # if has_phi:  # Phase-field weights are now mandatory.
    phi = np.asarray(geometry_data['phase_field'], dtype=np.float64)
    K = build_phase_diffusion_matrix(
        P_2d, neighbors, phi, geometry_data['phase_field_face_fraction'], Delta
    )
    substeps = diffusion_substeps(K, phi, dt)
    # Keep geometry weights and the implicit solve in double precision: small
    # occupied cells must not disappear through cancellation or mass flooring.
    L_gpu = cp_sparse.csr_matrix(K, dtype=cp.float64)
    mass_gpu = cp.asarray(phi)
    step_dt = float(dt) / substeps
    A_gpu = cp_sparse.diags(mass_gpu, format='csr') - (step_dt / 2) * L_gpu
    # Geometry and dt are fixed: one cached sparse product replaces the
    # per-substep mass multiply, diffusion product, scaling and addition.
    B_gpu = cp_sparse.diags(mass_gpu, format='csr') + (step_dt / 2) * L_gpu
    inverse_diagonal = 1.0 / A_gpu.diagonal()
    # Jacobi preconditioning is a pointwise multiply, not a sparse matvec.
    preconditioner = cp_linalg.LinearOperator(
        A_gpu.shape, matvec=lambda x: inverse_diagonal * x,
        rmatvec=lambda x: inverse_diagonal * x, dtype=A_gpu.dtype,
    )
    return dict(L=L_gpu, A=A_gpu, mass=mass_gpu, preconditioner=preconditioner,
                B=B_gpu, dt=step_dt, substeps=substeps, method='cg')


def crank_nicolson_diffusion_step_gpu(u_star_gpu, L_matrix_gpu, dt, method, A_gpu_cached,
                                    tol=1e-6, *, mass_gpu, preconditioner_gpu=None,
                                    rhs_matrix_gpu=None):
    # Solve implicit diffusion step using Crank-Nicolson on GPU
    
    # Solves: (M - dt/2 * L) * u_next = M*u_star + dt/2 * L * u_star.
    # M = diag(phi); an identity-mass fallback is no longer supported.
    
    # Args:
    #     u_star_gpu: CuPy array - voltage after reaction step
    #     L_matrix_gpu: CuPy sparse CSR matrix - diffusion operator
    #     dt: time step
    #     method: 'cg' (Conjugate Gradient) or 'gmres'
    #     A_gpu_cached: Pre-computed (M - dt/2 * L) matrix
    #     tol: tolerance for iterative solvers
    
    # Returns:
    #     u_next: CuPy array - voltage after diffusion step

    # Compute RHS: b = M*u_star + dt/2 * L * u_star
    if mass_gpu is None:
        raise ValueError("Phase-field diffusion requires mass_gpu (voxel-volume fractions)")
    if rhs_matrix_gpu is None:
        mass_u = mass_gpu * u_star_gpu
        b_gpu = mass_u + (dt / 2.0) * (L_matrix_gpu @ u_star_gpu)
    else:
        b_gpu = rhs_matrix_gpu @ u_star_gpu
    # solver_options = dict(atol=tol, maxiter=10000)  # Legacy tolerances.
    # if mass_gpu is not None:
    solver_options = dict(rtol=tol, atol=1e-7, maxiter=10000, M=preconditioner_gpu)
    
    # Solve linear system on GPU using available CuPy solvers
    if method == 'cg':
        u_next, info = cp_linalg.cg(A_gpu_cached, b_gpu, x0=u_star_gpu, **solver_options)
    elif method == 'gmres':
        u_next, info = cp_linalg.gmres(A_gpu_cached, b_gpu, x0=u_star_gpu, restart=50, **solver_options)
    
    # if mass_gpu is not None and info != 0:
    if info != 0:
        raise RuntimeError(f'Phase-field diffusion solver failed to converge (info={info})')
    
    return u_next

def compute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters):
    node_flag = arrhythmia_parameters['node_flag']
    temporary_block_voxel_id = []
    permanent_block_voxel_id = []
    if len(node_flag) > 0:
        temporary_block_voxel_id = np.where(node_flag == 3)[0] # temporary block for creating rotor or macro re-entry
        permanent_block_voxel_id = np.where(node_flag == 4)[0] # dense scar that are non-conductive

    # geometry data
    neighbor_id_2d = geometry_data['neighbor_id_2d']
    
    # simulation parameters
    t_final = float(simulation_parameters['t_final'])
    dt = simulation_parameters['dt']
    arrhythmia_flag = simulation_parameters['arrhythmia_flag']
    
    # pacing parameters
    J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t = assign_pacing_parameters(arrhythmia_parameters, arrhythmia_flag, n_voxel, neighbor_id_2d, simulation_parameters)
    
    # set initial value at rest
    if simulation_parameters['heart_model_flag'] == 0:
        u_current = np.zeros(n_voxel, dtype=np.float64)
        h_current = np.ones(n_voxel, dtype=np.float64)
    elif simulation_parameters['heart_model_flag'] == 1:
        u_current = np.zeros(n_voxel, dtype=np.float64)
        h_current = np.zeros(n_voxel, dtype=np.float64)
    
    # calculate the number of samples needed for 1 kHz sampling
    if simulation_parameters['heart_model_flag'] == 0:
        n_samples = int(np.round(t_final))
    elif simulation_parameters['heart_model_flag'] == 1:
        n_samples = int(np.round(t_final * simulation_parameters['time_scale']))
    
    sim_u_voxel = np.zeros((n_samples, n_voxel), dtype=np.float64)
    sim_h_voxel = np.zeros((n_samples, n_voxel), dtype=np.float64)
    physical_time = np.zeros(n_samples, dtype=np.float64)
    
    # GPU configuration
    threads_per_block = 256
    blocks_per_grid = (n_voxel + threads_per_block - 1) // threads_per_block
    
    # Convert to float64 and ensure contiguous arrays
    P_2d = np.ascontiguousarray(P_2d.astype(np.float64))
    J_stim = np.ascontiguousarray(J_stim.astype(np.float64))
    dt_float = np.float64(dt)
    
    # Permanent blocks are part of the baseline diffusion operator, so they
    # remain non-conductive for the whole simulation.
    P_2d_permanently_blocked = P_2d.copy()
    P_2d_permanently_blocked[permanent_block_voxel_id, 20] = 0.0

    # Build baseline diffusion matrix on GPU (only once).
    baseline_system = prepare_diffusion_system_gpu(
        P_2d_permanently_blocked, neighbor_id_2d, geometry_data, dt_float
    )

    # Pre-build the temporary-block matrix on top of the permanent blocks.
    if len(temporary_block_voxel_id) > 0:
        P_2d_blocked = P_2d_permanently_blocked.copy()
        P_2d_blocked[temporary_block_voxel_id, 20] = 0.0
        temporary_system = prepare_diffusion_system_gpu(
            P_2d_blocked, neighbor_id_2d, geometry_data, dt_float
        )
    else:
        temporary_system = baseline_system
    
    # Allocate GPU memory for CUDA kernels (CuPy arrays for unified memory access)
    d_u_current = cp.zeros(n_voxel, dtype=cp.float64)
    d_h_current = cp.zeros(n_voxel, dtype=cp.float64)
    d_u_star = cp.zeros(n_voxel, dtype=cp.float64)
    d_h_next = cp.zeros(n_voxel, dtype=cp.float64)
    d_P_2d = cp.asarray(P_2d)
    d_J_stim = cp.zeros(n_voxel, dtype=cp.float64)
    
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
        if ((model_time_step+1) % max(1, total_model_time_steps//5)) == 0:
            print(f'simulation {(model_time_step+1)/total_model_time_steps*100:.0f}%', end='\r')
        
        model_time = model_time_step * dt

        # temporary block for creating rotor or macro re-entry
        if model_time >= 0 and model_time < 200:
            active_system = temporary_system
        else:
            active_system = baseline_system
        
        # apply pacing - this is CPU-side since it has complex conditionals
        J_stim.fill(0.0)
        J_stim = apply_pacing(arrhythmia_parameters, simulation_parameters, arrhythmia_flag, model_time, J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, sim_u_voxel, sim_h_voxel, neighbor_id_2d)
        
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
        u_next = d_u_star
        for _ in range(active_system['substeps']):
            u_next = crank_nicolson_diffusion_step_gpu(
                u_next, active_system['L'], active_system['dt'],
                method=active_system['method'], A_gpu_cached=active_system['A'],
                # tol=1e-6 if active_system['mass'] is not None else 1e-5,
                tol=1e-6,
                mass_gpu=active_system['mass'],
                preconditioner_gpu=active_system['preconditioner'],
                rhs_matrix_gpu=active_system['B'],
            )
        
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
