"""CPU assembly for conservative, isotropic embedded-boundary diffusion."""

import numpy as np
from scipy import sparse

"""Static tissue-volume and face-area fractions on a Cartesian grid.

The tissue is a symmetric shell around a triangle surface, including rounded
rims at open mesh edges. This is a sharp embedded boundary, not a smoothed
tanh phase field. Only preprocessing samples are refined; voltage unknowns
remain on the original Cartesian spacing.
"""

from scipy.spatial import cKDTree


NEIGHBOR_OFFSETS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
], dtype=np.int64)


def _triangle_distance_squared(points, triangles):
    """Exact point-to-triangle distance for paired points and triangles."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum('ij,ij->i', normal, normal)
    height = np.einsum('ij,ij->i', ap, normal)
    projection = ap - (height / normal_sq)[:, None] * normal
    # Barycentric coordinates of the projected point.
    v = np.einsum('ij,ij->i', np.cross(projection, ac), normal) / normal_sq
    w = np.einsum('ij,ij->i', np.cross(ab, projection), normal) / normal_sq
    interior = (v >= 0) & (w >= 0) & (v + w <= 1)
    distance_sq = np.full(len(points), np.inf)
    distance_sq[interior] = height[interior] ** 2 / normal_sq[interior]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = np.einsum('ij,ij->i', points - start, edge)
        t /= np.einsum('ij,ij->i', edge, edge)
        difference = points - (start + np.clip(t, 0, 1)[:, None] * edge)
        distance_sq = np.minimum(distance_sq, np.einsum('ij,ij->i', difference, difference))
    return distance_sq


def _surface_distance(points, triangles, centers, radii, tree, search_radius):
    """Return inf beyond search_radius; bound temporary point/triangle arrays."""
    distances = np.full(len(points), np.inf)
    for first in range(0, len(points), 1024):
        batch = points[first:first + 1024]
        candidates = tree.query_ball_point(batch, search_radius + radii.max())
        counts = np.array([len(ids) for ids in candidates])
        if not counts.sum():
            continue
        point_ids = np.repeat(np.arange(len(batch)), counts)
        triangle_ids = np.concatenate(candidates).astype(np.int64)
        # The enclosing sphere of each triangle gives a conservative rejection.
        keep = np.linalg.norm(batch[point_ids] - centers[triangle_ids], axis=1)
        keep = keep <= search_radius + radii[triangle_ids]
        point_ids, triangle_ids = point_ids[keep], triangle_ids[keep]
        best = np.full(len(batch), np.inf)
        # A large triangle may make the tree query loose; also bound exact work.
        for offset in range(0, len(point_ids), 65536):
            p = point_ids[offset:offset + 65536]
            t = triangle_ids[offset:offset + 65536]
            np.minimum.at(best, p, _triangle_distance_squared(batch[p], triangles[t]))
        distances[first:first + len(batch)] = np.sqrt(best)
    return distances


def _inside_shell(points, triangles, centers, radii, tree, radius):
    """Test shell membership, refining only inconclusive nearest-center bounds.

    A hit on any triangle proves occupancy; finding the exact nearest triangle
    is unnecessary. Conversely, the nearest center distance minus the largest
    triangle radius is a lower bound on distance to the entire surface.
    """
    occupied = np.zeros(len(points), dtype=bool)
    if radius <= 0:
        return occupied
    max_radius = radii.max()
    for first in range(0, len(points), 65536):
        batch = points[first:first + 65536]
        center_distance, triangle_ids = tree.query(batch)
        possible = np.flatnonzero(center_distance <= radius + max_radius)
        if not len(possible):
            continue
        nearest_sq = _triangle_distance_squared(
            batch[possible], triangles[triangle_ids[possible]])
        # Use the same sqrt/strict comparison as the exact-distance path.
        hits = np.sqrt(nearest_sq) < radius
        occupied[first + possible[hits]] = True
        unresolved = possible[~hits]
        if len(unresolved):
            occupied[first + unresolved] = _surface_distance(
                batch[unresolved], triangles, centers, radii, tree, radius) < radius
    return occupied


def build_shell_geometry(vertex, face, Delta, thickness,
                         samples_per_axis=4, batch_size=256):
    """Return voxels, 18 neighbors, and static phase fractions for a mesh shell.

    ``thickness`` is the full shell width in voxel spacings, as in convert().
    Cell centers are at integer multiples of Delta. Midpoint quadrature measures
    occupied cell volumes and shared faces. Zero-volume sampled cells are omitted;
    positive fractions are never floored or smoothed. Repeat with more samples
    to assess geometric quadrature error, especially for small cuts and gaps.
    """
    vertex = np.asarray(vertex, dtype=np.float64)
    face = np.asarray(face)

    triangles = vertex[face]

    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centers)
    radius = thickness * Delta / 2

    def inside(points, limit=radius):
        return _inside_shell(points, triangles, centers, radii, tree, limit)

    q = ((np.arange(samples_per_axis) + 0.5) / samples_per_axis - 0.5) * Delta
    volume_offsets = np.stack(np.meshgrid(q, q, q, indexing='ij'), axis=-1).reshape(-1, 3)
    half_diagonal = np.sqrt(3) * Delta / 2
    low = np.floor((vertex.min(axis=0) - radius - Delta / 2) / Delta).astype(np.int64)
    high = np.ceil((vertex.max(axis=0) + radius + Delta / 2) / Delta).astype(np.int64)
    shape = high - low + 1
    grid_parts, phi_parts = [], []
    # Stream the bounding box; only allocate fine samples near the shell boundary.
    for first in range(0, int(np.prod(shape)), batch_size):
        print(f'Building shell geometry: {first/int(np.prod(shape))*100:.0f}%', end='\r')

        linear = np.arange(first, min(first + batch_size, int(np.prod(shape))))
        grid = np.column_stack(np.unravel_index(linear, tuple(shape))) + low
        xyz = grid * Delta
        keep = inside(xyz, radius + half_diagonal)
        grid, xyz = grid[keep], xyz[keep]
        phi = np.ones(len(grid))
        boundary = ~inside(xyz, radius - half_diagonal)
        if np.any(boundary):
            points = (xyz[boundary, None, :] + volume_offsets).reshape(-1, 3)
            occupied = inside(points)
            phi[boundary] = occupied.reshape(-1, len(volume_offsets)).mean(axis=1)
        active = phi > 0
        if np.any(active):
            grid_parts.append(grid[active])
            phi_parts.append(phi[active])

    grid = np.concatenate(grid_parts)
    phi = np.concatenate(phi_parts)
    voxel = grid * Delta
    grid_tree = cKDTree(grid)
    neighbors = np.full((len(grid), 18), -1, dtype=np.int64)
    for direction, offset in enumerate(NEIGHBOR_OFFSETS):
        dist, ids = grid_tree.query(grid + offset, distance_upper_bound=0.1)
        valid = np.isfinite(dist)
        neighbors[valid, direction] = ids[valid]

    face_fraction = np.zeros((len(grid), 6), dtype=np.float64)
    for axis in range(3):
        direction = 2 * axis
        ids = np.flatnonzero(neighbors[:, direction] >= 0)
        offsets = np.zeros((samples_per_axis ** 2, 3))
        tangents = [k for k in range(3) if k != axis]
        offsets[:, tangents] = np.stack(np.meshgrid(q, q, indexing='ij'), axis=-1).reshape(-1, 2)
        offsets[:, axis] = Delta / 2
        for first in range(0, len(ids), batch_size):
            i = ids[first:first + batch_size]
            j = neighbors[i, direction]
            points = (voxel[i, None, :] + offsets).reshape(-1, 3)
            area = inside(points).reshape(len(i), -1).mean(axis=1)
            face_fraction[i, direction] = area
            face_fraction[j, direction + 1] = area

    return dict(voxel=voxel, neighbor_id_2d=neighbors, Delta=float(Delta),
                phase_field=phi.astype(np.float64),
                phase_field_face_fraction=face_fraction.astype(np.float64),
                phase_field_thickness_mm=float(thickness * Delta),
                phase_field_samples_per_axis=int(samples_per_axis),
                phase_field_method='shell_volume_fraction_v1')



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