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
from numba import njit, prange
from scipy import sparse
from scipy.sparse.linalg import bicgstab
from modules.pacing import assign_pacing_parameters, apply_pacing

def build_diffusion_matrix(P_2d, neighbor_id_2d_2, Delta):
    n_voxel = P_2d.shape[0]
    
    # Lists to build sparse matrix in COO format
    row_indices = []
    col_indices = []
    data_values = []
    
    for n in range(n_voxel):
        # Diffusion coefficient
        D_coeff = P_2d[n, 20] / (4 * Delta**2)
        
        # Diagonal entry (central node contribution)
        diagonal_sum = -(P_2d[n, 0] + P_2d[n, 1] + P_2d[n, 2] + 
                        P_2d[n, 3] + P_2d[n, 4] + P_2d[n, 5])
        row_indices.append(n)
        col_indices.append(n)
        data_values.append(D_coeff * diagonal_sum)
        
        # Off-diagonal entries (neighbor contributions)
        # Direct neighbors (6 main directions)
        neighbor_coeffs = [
            (neighbor_id_2d_2[n, 0], P_2d[n, 0]),
            (neighbor_id_2d_2[n, 1], P_2d[n, 1]),
            (neighbor_id_2d_2[n, 2], P_2d[n, 2]),
            (neighbor_id_2d_2[n, 3], P_2d[n, 3]),
            (neighbor_id_2d_2[n, 4], P_2d[n, 4]),
            (neighbor_id_2d_2[n, 5], P_2d[n, 5]),
        ]
        
        for neighbor_id, coeff in neighbor_coeffs:
            if neighbor_id >= 0 and neighbor_id < n_voxel:  # Valid neighbor
                row_indices.append(n)
                col_indices.append(neighbor_id)
                data_values.append(D_coeff * coeff)
        
        # Cross-derivative terms (anisotropy)
        # P_2d[n, 6]: u[neighbor 0] - u[neighbor 1]
        if neighbor_id_2d_2[n, 0] >= 0 and neighbor_id_2d_2[n, 0] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 0])
            data_values.append(D_coeff * P_2d[n, 6])
        if neighbor_id_2d_2[n, 1] >= 0 and neighbor_id_2d_2[n, 1] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 1])
            data_values.append(-D_coeff * P_2d[n, 6])
        
        # P_2d[n, 7]: u[neighbor 2] - u[neighbor 3]
        if neighbor_id_2d_2[n, 2] >= 0 and neighbor_id_2d_2[n, 2] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 2])
            data_values.append(D_coeff * P_2d[n, 7])
        if neighbor_id_2d_2[n, 3] >= 0 and neighbor_id_2d_2[n, 3] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 3])
            data_values.append(-D_coeff * P_2d[n, 7])
        
        # P_2d[n, 8]: u[neighbor 4] - u[neighbor 5]
        if neighbor_id_2d_2[n, 4] >= 0 and neighbor_id_2d_2[n, 4] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 4])
            data_values.append(D_coeff * P_2d[n, 8])
        if neighbor_id_2d_2[n, 5] >= 0 and neighbor_id_2d_2[n, 5] < n_voxel:
            row_indices.append(n)
            col_indices.append(neighbor_id_2d_2[n, 5])
            data_values.append(-D_coeff * P_2d[n, 8])
        
        # Additional cross terms (edges and corners)
        cross_terms = [
            (neighbor_id_2d_2[n, 6], neighbor_id_2d_2[n, 8], P_2d[n, 9]),   # 6-8
            (neighbor_id_2d_2[n, 9], neighbor_id_2d_2[n, 7], P_2d[n, 10]),  # 9-7
            (neighbor_id_2d_2[n, 14], neighbor_id_2d_2[n, 16], P_2d[n, 11]), # 14-16
            (neighbor_id_2d_2[n, 17], neighbor_id_2d_2[n, 15], P_2d[n, 12]), # 17-15
            (neighbor_id_2d_2[n, 10], neighbor_id_2d_2[n, 12], P_2d[n, 13]), # 10-12
            (neighbor_id_2d_2[n, 13], neighbor_id_2d_2[n, 11], P_2d[n, 14]), # 13-11
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
    
    # Build sparse matrix in COO format, then convert to CSR
    L = sparse.coo_matrix((data_values, (row_indices, col_indices)), 
                          shape=(n_voxel, n_voxel))
    return L.tocsr()

def crank_nicolson_diffusion_step(u_star, L_matrix, dt, tol=1e-6):
    """
    Solve implicit diffusion step using Crank-Nicolson
    
    Solves: (I - dt/2 * L) * u_next = u_star + dt/2 * L * u_star
    
    Args:
        u_star: voltage after reaction step
        L_matrix: diffusion operator (sparse matrix)
        dt: time step
        method: 'direct', 'cg' (Conjugate Gradient), or 'bicgstab'
        tol: tolerance for iterative solvers
    
    Returns:
        u_next: voltage after diffusion step
    """

    n_voxel = len(u_star)
    I = sparse.identity(n_voxel, format='csr')
    
    # Build system: A * u_next = b
    A = I - (dt / 2.0) * L_matrix
    b = u_star + (dt / 2.0) * (L_matrix @ u_star)
    
    # Solve linear system
    u_next, info = bicgstab(A, b, x0=u_star, atol=tol, maxiter=1000)
    if info != 0:
        print(f"Warning: BiCGSTAB did not converge (info={info})")
    
    return u_next

@njit
def compute_reaction_only(heart_model_flag, u, h, P_2d, J_stim, n):
    if heart_model_flag == 0:  # Mitchell-Schaeffer model
        du_dt = (h[n] * (u[n]**2) * (1 - u[n])) / P_2d[n, 17] - \
                u[n] / P_2d[n, 18] + J_stim[n]
        
        if u[n] < P_2d[n, 19]:
            dh_dt = (1 - h[n]) / P_2d[n, 15]
        else:
            dh_dt = -h[n] / P_2d[n, 16]
            
    elif heart_model_flag == 1:  # Aliev-Panfilov model
        du_dt = -P_2d[n, 15]*u[n]*(u[n] - P_2d[n, 16])*(u[n] - 1) - u[n]*h[n] + J_stim[n]
        dh_dt = (P_2d[n, 17] + P_2d[n, 18]*h[n]/(u[n] + P_2d[n, 19])) * \
                (-h[n] - P_2d[n, 15]*u[n]*(u[n] - P_2d[n, 16] - 1))
    
    return du_dt, dh_dt

@njit(parallel=True)
def reaction_step_rk4(heart_model_flag, u_current, h_current, P_2d, J_stim, dt):
    n_voxel = u_current.shape[0]
    u_next = np.empty_like(u_current)
    h_next = np.empty_like(h_current)
    
    k1_u = np.zeros(n_voxel)
    k1_h = np.zeros(n_voxel)
    k2_u = np.zeros(n_voxel)
    k2_h = np.zeros(n_voxel)
    k3_u = np.zeros(n_voxel)
    k3_h = np.zeros(n_voxel)
    k4_u = np.zeros(n_voxel)
    k4_h = np.zeros(n_voxel)
    u_temp = np.empty_like(u_current)
    h_temp = np.empty_like(h_current)
    
    # K1
    for n in prange(n_voxel):
        k1_u[n], k1_h[n] = compute_reaction_only(heart_model_flag, u_current, h_current, P_2d, J_stim, n)
    
    # K2
    for n in prange(n_voxel):
        u_temp[n] = u_current[n] + 0.5 * dt * k1_u[n]
        h_temp[n] = h_current[n] + 0.5 * dt * k1_h[n]
    for n in prange(n_voxel):
        k2_u[n], k2_h[n] = compute_reaction_only(heart_model_flag, u_temp, h_temp, P_2d, J_stim, n)
    
    # K3
    for n in prange(n_voxel):
        u_temp[n] = u_current[n] + 0.5 * dt * k2_u[n]
        h_temp[n] = h_current[n] + 0.5 * dt * k2_h[n]
    for n in prange(n_voxel):
        k3_u[n], k3_h[n] = compute_reaction_only(heart_model_flag, u_temp, h_temp, P_2d, J_stim, n)
    
    # K4
    for n in prange(n_voxel):
        u_temp[n] = u_current[n] + dt * k3_u[n]
        h_temp[n] = h_current[n] + dt * k3_h[n]
    for n in prange(n_voxel):
        k4_u[n], k4_h[n] = compute_reaction_only(heart_model_flag, u_temp, h_temp, P_2d, J_stim, n)
    
    # Combine
    for n in prange(n_voxel):
        u_next[n] = u_current[n] + (dt / 6.0) * (k1_u[n] + 2*k2_u[n] + 2*k3_u[n] + k4_u[n])
        h_next[n] = h_current[n] + (dt / 6.0) * (k1_h[n] + 2*k2_h[n] + 2*k3_h[n] + k4_h[n])
    
    return u_next, h_next

def compute_voxel_crank_nicolson(heart_model_flag, u_current, h_current, P_2d, 
                                  neighbor_id_2d_2, J_stim, dt, Delta, L_matrix, solver_method='bicgstab'):
    """
    Operator splitting: Reaction (RK4) + Diffusion (Crank-Nicolson)
    
    Step 1: u*, h* = u_current + dt * Reaction(u_current, h_current)
    Step 2: u_next = Crank-Nicolson diffusion on u*
    
    Returns: u_next, h_next
    """

    # Step 1: Reaction step (explicit RK4)
    u_star, h_next = reaction_step_rk4(heart_model_flag, u_current, h_current, P_2d, J_stim, dt)
    
    # Step 2: Diffusion step (implicit Crank-Nicolson)
    u_next = crank_nicolson_diffusion_step(u_star, L_matrix, dt)
    
    return u_next, h_next

def execute(n_voxel, P_2d, geometry_data, simulation_parameters, arrhythmia_parameters):    
    # geometry data
    neighbor_id_2d = geometry_data['neighbor_id_2d']
    Delta = geometry_data['Delta']

    # simulation parameters
    t_final = int(simulation_parameters['t_final'])
    dt = simulation_parameters['dt']
    arrhythmia_flag = simulation_parameters['arrhythmia_flag']
    
    # Crank-Nicolson solver method
    cn_solver = simulation_parameters.get('cn_solver_method', 'bicgstab')  # 'direct', 'cg', or 'bicgstab'
    
    # pacing parameters
    J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, ap_min, ap_max, h_min, h_max, s2_region_size_factor = assign_pacing_parameters(arrhythmia_parameters, arrhythmia_flag, n_voxel, neighbor_id_2d, simulation_parameters)

    # set initial value at rest
    if simulation_parameters['heart_model_flag'] == 0:
        u_current = np.zeros(n_voxel)
        h_current = np.ones(n_voxel)
    elif simulation_parameters['heart_model_flag'] == 1:
        u_current = np.zeros(n_voxel)
        h_current = np.zeros(n_voxel)

    # Build diffusion matrix once (doesn't change during simulation)
    neighbor_id_2d_2 = neighbor_id_2d.copy()
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0
    L_matrix = build_diffusion_matrix(P_2d, neighbor_id_2d_2, Delta)

    id_save = 0

    # calculate the number of samples needed for 1 kHz sampling
    if simulation_parameters['heart_model_flag'] == 0:
        n_samples = t_final
    elif simulation_parameters['heart_model_flag'] == 1:
        n_samples = int(t_final * simulation_parameters['time_scale'])

    sim_u_voxel = np.zeros((n_samples, n_voxel))
    sim_h_voxel = np.zeros((n_samples, n_voxel))
    physical_time = np.zeros(n_samples)

    total_model_time_steps = int(t_final/dt)
    for model_time_step in range(total_model_time_steps):
        if ((model_time_step+1) % (total_model_time_steps//5)) == 0:
            print(f'simulating {(model_time_step+1)/total_model_time_steps*100:.1f}%')
        
        model_time = model_time_step * dt

        # apply pacing
        J_stim.fill(0.0)
        J_stim = apply_pacing(arrhythmia_parameters, simulation_parameters, arrhythmia_flag, model_time, J_stim, s1_pacing_voxel_id, s2_pacing_voxel_id, s1_t, J_stim_magnitude, pacing_duration, s2_t, ap_min, ap_max, h_min, h_max, s2_region_size_factor, sim_u_voxel, sim_h_voxel, neighbor_id_2d)

        # update value using Crank-Nicolson
        u_next, h_next = compute_voxel_crank_nicolson(
            simulation_parameters['heart_model_flag'], 
            u_current, h_current, P_2d, neighbor_id_2d_2, 
            J_stim, dt, Delta, L_matrix, solver_method=cn_solver
        )
        u_current = u_next
        h_current = h_next
        
        # save value at 1 kHz (every 1 ms)
        number_of_steps_per_ms = int(1 / (dt * simulation_parameters['time_scale']))
        if number_of_steps_per_ms > 0:
            if (model_time_step % number_of_steps_per_ms) == 0 and id_save < n_samples:
                sim_u_voxel[id_save, :] = u_current
                sim_h_voxel[id_save, :] = h_current
                physical_time[id_save] = model_time * simulation_parameters['time_scale']
                id_save = id_save + 1
        else:
            # If dt is too large, save every step
            if id_save < n_samples:
                sim_u_voxel[id_save, :] = u_current
                sim_h_voxel[id_save, :] = h_current
                physical_time[id_save] = model_time * simulation_parameters['time_scale']
                id_save = id_save + 1

    # if dt > 1, it might not have filled all samples, delete the zeros at the end
    while sim_u_voxel[-1, 0] == 0:
        sim_u_voxel = sim_u_voxel[:-1, :]
        sim_h_voxel = sim_h_voxel[:-1, :]
        physical_time = physical_time[:-1]

    return sim_u_voxel, sim_h_voxel, physical_time
