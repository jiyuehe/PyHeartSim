import numpy as np

def extract_diffusion_tensor_components(D0, n_voxel, dtype=np.float64):
    D11 = np.zeros(n_voxel, dtype=dtype)
    D12 = np.zeros(n_voxel, dtype=dtype)
    D13 = np.zeros(n_voxel, dtype=dtype)
    D21 = np.zeros(n_voxel, dtype=dtype)
    D22 = np.zeros(n_voxel, dtype=dtype)
    D23 = np.zeros(n_voxel, dtype=dtype)
    D31 = np.zeros(n_voxel, dtype=dtype)
    D32 = np.zeros(n_voxel, dtype=dtype)
    D33 = np.zeros(n_voxel, dtype=dtype)
    
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
    
    return D11, D12, D13, D21, D22, D23, D31, D32, D33

def compute_electrode_distances(electrode_xyz, voxel, dtype=np.float64):
    n_voxel = voxel.shape[0]
    n_electrode = electrode_xyz.shape[0]
    
    l = np.zeros((n_voxel, n_electrode), dtype=dtype)
    l_x = np.zeros((n_voxel, n_electrode), dtype=dtype)
    l_y = np.zeros((n_voxel, n_electrode), dtype=dtype)
    l_z = np.zeros((n_voxel, n_electrode), dtype=dtype)
    
    for e_id in range(n_electrode):
        l_x[:, e_id] = voxel[:, 0] - electrode_xyz[e_id, 0]
        l_y[:, e_id] = voxel[:, 1] - electrode_xyz[e_id, 1]
        l_z[:, e_id] = voxel[:, 2] - electrode_xyz[e_id, 2]
        l[:, e_id] = np.sqrt(l_x[:, e_id]**2 + l_y[:, e_id]**2 + l_z[:, e_id]**2)
    
    l[l < 1] = 1  # electrode is at least 1 mm away from tissue
    
    return l, l_x, l_y, l_z

def compute_gradients(action_potential, neighbor_id_2d, Delta):
    neighbor_id_2d_copy = neighbor_id_2d.copy()
    
    # x direction
    px = neighbor_id_2d_copy[:, 0]
    px_0_id = px == -1
    px[px_0_id] = 0
    
    mx = neighbor_id_2d_copy[:, 1]
    mx_0_id = mx == -1
    mx[mx_0_id] = 0
    
    # y direction
    py = neighbor_id_2d_copy[:, 2]
    py_0_id = py == -1
    py[py_0_id] = 0
    
    my = neighbor_id_2d_copy[:, 3]
    my_0_id = my == -1
    my[my_0_id] = 0
    
    # z direction
    pz = neighbor_id_2d_copy[:, 4]
    pz_0_id = pz == -1
    pz[pz_0_id] = 0
    
    mz = neighbor_id_2d_copy[:, 5]
    mz_0_id = mz == -1
    mz[mz_0_id] = 0
    
    # Compute gradients
    dvdx = (action_potential[:, px] - action_potential[:, mx]) / (2 * Delta)
    dvdx[:, px_0_id] = 0
    dvdx[:, mx_0_id] = 0
    
    dvdy = (action_potential[:, py] - action_potential[:, my]) / (2 * Delta)
    dvdy[:, py_0_id] = 0
    dvdy[:, my_0_id] = 0
    
    dvdz = (action_potential[:, pz] - action_potential[:, mz]) / (2 * Delta)
    dvdz[:, pz_0_id] = 0
    dvdz[:, mz_0_id] = 0
    
    return dvdx, dvdy, dvdz

def broadcast_diffusion_tensors(D11, D12, D13, D21, D22, D23, D31, D32, D33, n_electrode, dtype=np.float64):
    D11_b = np.tile(D11[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D12_b = np.tile(D12[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D13_b = np.tile(D13[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D21_b = np.tile(D21[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D22_b = np.tile(D22[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D23_b = np.tile(D23[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D31_b = np.tile(D31[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D32_b = np.tile(D32[:, np.newaxis], (1, n_electrode)).astype(dtype)
    D33_b = np.tile(D33[:, np.newaxis], (1, n_electrode)).astype(dtype)
    
    return D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b

def normalize_electrogram_magnitude(electrogram_unipolar, typical_magnitude=1.0):
    n_electrode = electrogram_unipolar.shape[1]
    voltage = np.array([np.ptp(electrogram_unipolar[:, n]) for n in range(n_electrode)])
    voltage_mean = np.mean(voltage)
    scale = typical_magnitude / voltage_mean
    
    return electrogram_unipolar * scale
