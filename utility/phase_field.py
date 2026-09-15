"""Static tissue-volume and face-area fractions on a Cartesian grid.

The tissue is a symmetric shell around a triangle surface, including rounded
rims at open mesh edges. This is a sharp embedded boundary, not a smoothed
tanh phase field. Only preprocessing samples are refined; voltage unknowns
remain on the original Cartesian spacing.
"""

import numpy as np
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
    if not grid_parts:
        raise ValueError('No occupied voxels were sampled; check mesh units, thickness and spacing')
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
