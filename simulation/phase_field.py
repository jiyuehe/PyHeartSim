"""CPU assembly for conservative, isotropic embedded-boundary diffusion."""

import numpy as np
from scipy import sparse


def build_diffusion_matrix(P_2d, neighbors, phi, face_fraction, Delta):
    """Assemble K for diag(phi) du/dt = K u using shared face conductances.

    Only D0=I is supported. P_2d[:, 20] is treated as physical scalar diffusion
    inside the flux; its harmonic face mean is zero if either cell is blocked.
    This intentionally differs from legacy row scaling for heterogeneous c.
    """
    P_2d = np.asarray(P_2d)
    neighbors = np.asarray(neighbors)
    phi = np.asarray(phi, dtype=np.float64)
    fractions = np.asarray(face_fraction, dtype=np.float64)
    n = len(phi)
    if phi.shape != (n,) or n == 0 or np.any(~np.isfinite(phi)) or np.any((phi <= 0) | (phi > 1)):
        raise ValueError('phase_field must contain one finite fraction in (0, 1] per voxel')
    if P_2d.shape != (n, 21) or neighbors.shape != (n, 18) or fractions.shape != (n, 6):
        raise ValueError('Phase geometry, neighbors and P_2d must have matching voxel counts')
    if (not np.issubdtype(neighbors.dtype, np.integer)
            or np.any((neighbors < -1) | (neighbors >= n))):
        raise ValueError('Neighbor indices must be -1 or valid voxel indices')
    if not np.isfinite(Delta) or Delta <= 0:
        raise ValueError('Delta must be positive and finite')
    if np.any(~np.isfinite(fractions)) or np.any((fractions < 0) | (fractions > 1)):
        raise ValueError('Face fractions must be finite and in [0, 1]')
    direct = neighbors[:, :6]
    if np.any(fractions[direct < 0] != 0):
        raise ValueError('Missing neighbors must have zero face fraction')
    expected = 4.0 * (direct >= 0)
    if not np.allclose(P_2d[:, :6], expected) or not np.allclose(P_2d[:, 6:15], 0):
        raise ValueError('Phase-field diffusion currently requires isotropic D0=I; tensor diffusion is unsupported')
    c = np.asarray(P_2d[:, 20], dtype=np.float64)
    if np.any(~np.isfinite(c)) or np.any(c < 0):
        raise ValueError('Diffusion coefficients must be nonnegative and finite')

    rows, cols, values = [], [], []
    for direction in range(6):
        i = np.flatnonzero(direct[:, direction] >= 0)
        j = direct[i, direction]
        opposite = direction ^ 1
        if np.any(i == j) or np.any(direct[j, opposite] != i):
            raise ValueError('Face neighbors must be reciprocal and cannot reference themselves')
        if not np.allclose(fractions[i, direction], fractions[j, opposite], rtol=0, atol=1e-7):
            raise ValueError('Both sides of a shared face must have the same fraction')
        if direction % 2:
            continue
        conductance = np.zeros(len(i))
        conducting = (c[i] > 0) & (c[j] > 0)
        # Stable harmonic mean, including exact insulating blocks.
        lo = np.minimum(c[i[conducting]], c[j[conducting]])
        hi = np.maximum(c[i[conducting]], c[j[conducting]])
        conductance[conducting] = 2 * lo / (1 + lo / hi)
        conductance *= fractions[i, direction] / Delta ** 2
        rows.extend((i, j, i, j))
        cols.extend((j, i, i, j))
        values.extend((conductance, conductance, -conductance, -conductance))
    K = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()
    K.eliminate_zeros()
    return K


def diffusion_substeps(K, phi, dt):
    # """Keep the CN RHS nonnegative without changing tiny cut-cell volumes.

    # dt_sub * sum_j(g_ij) <= 2 phi_i makes both CN factors positivity preserving
    # for this scalar face-flux operator (up to iterative-solver error).
    # """
    # if not np.isfinite(dt) or dt <= 0:
    #     raise ValueError('dt must be positive and finite')
    # rate = np.max(-K.diagonal() / np.asarray(phi))
    # return max(1, int(np.ceil(dt * rate / 2)))
    return 1