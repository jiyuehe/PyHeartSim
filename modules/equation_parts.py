import numpy as np

def load_fiber(n_voxel):
    # fiber orientations
    fiber_flag = 0 # 0: no fiber, 1: fiber

    if fiber_flag == 0:
        r = [] # no fiber
        fiber_orientation = [] # no fiber

    D0 = [None] * n_voxel  # Create list of None values (equivalent to cell array)
    for n in range(n_voxel):  # 0-based indexing in Python
        if fiber_flag == 1:
            e1 = fiber_orientation[n, :].reshape(-1, 1)  # Make column vector
            D0[n] = r * np.eye(3) + (1-r) * (e1 @ e1.T)  # @ is matrix multiplication
        elif fiber_flag == 0:
            # here r = 1
            D0[n] = np.eye(3)

    return D0

def heart_model_equation_parts(simulation_parameters, n_voxel, D0, neighbor_id_2d, parameter):
    D11 = np.zeros(n_voxel)
    D12 = np.zeros(n_voxel)
    D13 = np.zeros(n_voxel)
    D21 = np.zeros(n_voxel)
    D22 = np.zeros(n_voxel)
    D23 = np.zeros(n_voxel)
    D31 = np.zeros(n_voxel)
    D32 = np.zeros(n_voxel)
    D33 = np.zeros(n_voxel)
    for n in range(n_voxel):
        D11[n] = D0[n][0, 0]
        D12[n] = D0[n][0, 1]
        D13[n] = D0[n][0, 2]
        D21[n] = D0[n][1, 0]
        D22[n] = D0[n][1, 1]
        D23[n] = D0[n][1, 2]
        D31[n] = D0[n][2, 0]
        D32[n] = D0[n][2, 1]
        D33[n] = D0[n][2, 2]

    delta_2d = np.sign(neighbor_id_2d + 1) # it will result in 0s and 1s. 
        # neighbor_id_2d contains -1 for nodes have no neighbors
        # note: for those 0s in neighbor_id_2d, it means the neighbor is the first voxel (index 0)
        # delta_2d = 1 for nodes have neighbors
        # delta_2d = 0 for nodes have no neighbors

    P_2d = np.zeros((n_voxel, 21))  # parts of the equation. _2d signifies it is a 2D variable

    P_2d[:, 0] = 4 * delta_2d[:, 0] * D11
    P_2d[:, 1] = 4 * delta_2d[:, 1] * D11
    P_2d[:, 2] = 4 * delta_2d[:, 2] * D22
    P_2d[:, 3] = 4 * delta_2d[:, 3] * D22
    P_2d[:, 4] = 4 * delta_2d[:, 4] * D33
    P_2d[:, 5] = 4 * delta_2d[:, 5] * D33

    neighbor_id_2d_2 = neighbor_id_2d.copy() # NOTE: without .copy(), changes of neighbor_id_2d_2 will also change neighbor_id_2d
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0 # change -1 to 0, so that it can be used as index. 
        # and this does not matter because the corresponding delta_2d will be 0, 
        # so these terms will be eliminated anyway

    P_2d[:, 6] = (delta_2d[:, 0] * delta_2d[:, 1] * 
                (delta_2d[:, 0] * delta_2d[:, 1] * (D11[neighbor_id_2d_2[:, 0]] - D11[neighbor_id_2d_2[:, 1]]) +
                delta_2d[:, 2] * delta_2d[:, 3] * (D21[neighbor_id_2d_2[:, 2]] - D21[neighbor_id_2d_2[:, 3]]) +
                delta_2d[:, 4] * delta_2d[:, 5] * (D31[neighbor_id_2d_2[:, 4]] - D31[neighbor_id_2d_2[:, 5]])))

    P_2d[:, 7] = (delta_2d[:, 2] * delta_2d[:, 3] * 
                (delta_2d[:, 0] * delta_2d[:, 1] * (D12[neighbor_id_2d_2[:, 0]] - D12[neighbor_id_2d_2[:, 1]]) +
                delta_2d[:, 2] * delta_2d[:, 3] * (D22[neighbor_id_2d_2[:, 2]] - D22[neighbor_id_2d_2[:, 3]]) +
                delta_2d[:, 4] * delta_2d[:, 5] * (D32[neighbor_id_2d_2[:, 4]] - D32[neighbor_id_2d_2[:, 5]])))

    P_2d[:, 8] = (delta_2d[:, 4] * delta_2d[:, 5] * 
                (delta_2d[:, 0] * delta_2d[:, 1] * (D13[neighbor_id_2d_2[:, 0]] - D13[neighbor_id_2d_2[:, 1]]) +
                delta_2d[:, 2] * delta_2d[:, 3] * (D23[neighbor_id_2d_2[:, 2]] - D23[neighbor_id_2d_2[:, 3]]) +
                delta_2d[:, 4] * delta_2d[:, 5] * (D33[neighbor_id_2d_2[:, 4]] - D33[neighbor_id_2d_2[:, 5]])))

    P_2d[:, 9] = 2 * delta_2d[:, 6] * delta_2d[:, 8] * D12
    P_2d[:, 10] = 2 * delta_2d[:, 7] * delta_2d[:, 9] * D12
    P_2d[:, 11] = 2 * delta_2d[:, 14] * delta_2d[:, 16] * D13
    P_2d[:, 12] = 2 * delta_2d[:, 15] * delta_2d[:, 17] * D13
    P_2d[:, 13] = 2 * delta_2d[:, 10] * delta_2d[:, 12] * D23
    P_2d[:, 14] = 2 * delta_2d[:, 11] * delta_2d[:, 13] * D23

    if simulation_parameters['heart_model_flag'] == 0: # Mitchell-Schaeffer model
        P_2d[:, 15] = parameter['tau_open_voxel']
        P_2d[:, 16] = parameter['tau_close_voxel']
        P_2d[:, 17] = parameter['tau_in_voxel']
        P_2d[:, 18] = parameter['tau_out_voxel']
        P_2d[:, 19] = parameter['v_gate_voxel']
    elif simulation_parameters['heart_model_flag'] == 1: # Aliev-Panfilov model
        P_2d[:, 15] = parameter['k_voxel']
        P_2d[:, 16] = parameter['a_voxel']
        P_2d[:, 17] = parameter['epsilon_0_voxel']
        P_2d[:, 18] = parameter['mu1_voxel']
        P_2d[:, 19] = parameter['mu2_voxel']
    
    P_2d[:, 20] = parameter['c_voxel']

    return P_2d
