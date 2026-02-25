import numpy as np

def execute(n_voxel):
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
