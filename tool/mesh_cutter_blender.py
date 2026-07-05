# Copyright 2026 Mason Manetta
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

# How cuts work
# -------------
# Each cutter object (cube or cylinder) is a convex mesh.  Its outward-facing
# face planes are extracted from the cutter's mesh data and transformed into the
# target mesh's local space.  We then:
#   1. Bisect the target bmesh along every plane (no geometry is removed yet;
#      we just create new vertices/edges at each intersection).
#   2. Delete every face whose centroid lies strictly inside the convex hull
#      of the cutter planes.
# Cuts are stored as a bmesh snapshot so toggling/restoring is a simple memcpy
# rather than a modifier rebuild.
#
# NOTE: This approach assumes each cutter is a *convex* mesh.  Non-convex
# cutters will produce unexpected results.

# Instructions for use
# --------------------
# Press k to toggle Cut Mode
# Press s to save the obj

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

#%%

ENABLE_COMMON_DEBUG_IMPORTS = False
debug_plot = 0
if ENABLE_COMMON_DEBUG_IMPORTS:
    import common

if debug_plot == 1:
    if not ENABLE_COMMON_DEBUG_IMPORTS:
        raise RuntimeError("Set ENABLE_COMMON_DEBUG_IMPORTS = True to use debug_plot with common.py")

    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"

    file_dir = Path('/home/j/Desktop/hdd/share_folder/patient_data')
    mesh_name = '99_2-LaFAM_cartofinder_data'
    vertex, face = common.load_obj(file_dir, mesh_name+'_refined')

    # Tip detection is performed directly from the loaded mesh by
    # MESH_OT_KnifeCutter._identify_tip_regions(); no *_tip_vertex.txt file is
    # required for automatic cutter placement.
    tip_vertex = np.empty((0, 3))
    center_of_mass = np.mean(vertex, axis=0)

    # build face list for Mesh3d
    fi, fj, fk = face[:, 0], face[:, 1], face[:, 2]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
            i=fi, j=fj, k=fk,
            color='white', opacity=0.2,
            lighting=dict(ambient=1.0, diffuse=0, specular=0, roughness=1, fresnel=0),
            flatshading=True,
        ),

        # triangle edges: interleave [A, B, C, A, None] per triangle
        go.Scatter3d(
            x=np.stack([vertex[fi, 0], vertex[fj, 0], vertex[fk, 0], vertex[fi, 0], np.full(len(fi), None)], axis=1).ravel(),
            y=np.stack([vertex[fi, 1], vertex[fj, 1], vertex[fk, 1], vertex[fi, 1], np.full(len(fi), None)], axis=1).ravel(),
            z=np.stack([vertex[fi, 2], vertex[fj, 2], vertex[fk, 2], vertex[fi, 2], np.full(len(fi), None)], axis=1).ravel(),
            mode='lines',
            opacity=0.5,
            line=dict(color='gray', width=1),
        ),

        # detected tip vertices as large red dots, when populated for debugging
        go.Scatter3d(
            x=tip_vertex[:, 0], y=tip_vertex[:, 1], z=tip_vertex[:, 2],
            mode='markers',
            marker=dict(size=8, color='red'),
        ),

        # center of mass as a black dot
        go.Scatter3d(
            x=[center_of_mass[0]], y=[center_of_mass[1]], z=[center_of_mass[2]],
            mode='markers',
            marker=dict(size=8, color='black'),
        ),

        # lines from each tip vertex to the center of mass
        go.Scatter3d(
            x=np.stack([tip_vertex[:, 0], np.full(len(tip_vertex), center_of_mass[0]), np.full(len(tip_vertex), None)], axis=1).ravel(),
            y=np.stack([tip_vertex[:, 1], np.full(len(tip_vertex), center_of_mass[1]), np.full(len(tip_vertex), None)], axis=1).ravel(),
            z=np.stack([tip_vertex[:, 2], np.full(len(tip_vertex), center_of_mass[2]), np.full(len(tip_vertex), None)], axis=1).ravel(),
            mode='lines',
            line=dict(color='black', width=3),
        ),
    ])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show() # opens in browser

#%%
import bpy
import bmesh
import os
import ast
from math import radians
from mathutils import Vector

# --- CONFIGURATION ---
name_prefix = '99_2-LaFAM_cartofinder_data'

BASE_PATH = Path("//")
#BASE_PATH = Path("/home/mason/Code/PyHeartSim/")

FILE_PATH = BASE_PATH / f"{name_prefix}_refined.obj"
EXPORT_FILE_PATH = BASE_PATH / f"{name_prefix}_refined_cut.obj"
CUTS_FILE_PATH = BASE_PATH / f"{name_prefix}_cuts.yaml"
USE_SCRIPT_DIR_FALLBACK = True
LOAD_CUTS = True
SHOW_TIP_VERTEX_MARKERS = True
AUTO_CREATE_VEIN_CUTTERS = True

# Number of additional interactive cutters to create.
N_EXTRA_CUT_CUBES = 0
N_EXTRA_CUT_CYLINDERS = 0
EXTRA_CYLINDER_DEFAULT_LOCATION = (38.672, -59.443, 6.6735)
EXTRA_CYLINDER_DEFAULT_ROTATION_DEG = (84.724, -10.902, 35.284)
EXTRA_CYLINDER_DEFAULT_SCALE = (13.928, 13.928, 40.965)

# Cutter sizing, expressed as a fraction of average target dimension.
CUBE_SCALE_FACTOR = 0.3
CYLINDER_RADIUS_FACTOR = 0.17
CYLINDER_DEPTH_FACTOR = 0.5

# Additional spacing so cutters spawn outside the target mesh bounds.
SPAWN_MARGIN_FACTOR = 0.20

# Relative epsilon for plane-distance tests and bisect snapping.  Applied as a
# fraction of the target mesh's average dimension so it scales with the model.
BISECT_EPSILON_FACTOR = 1e-5
TIP_MARKER_RADIUS_FACTOR = 0.015
TIP_CLUSTER_DISTANCE = 25.0
N_PULMONARY_VEIN_TIPS = 4
VEIN_CUT_OFFSET_MM = 10.0
VEIN_CUT_PLANE_WINDOW_MM = 4.0
VEIN_CUT_RADIUS_MARGIN_MM = 2.0
VEIN_CUT_RADIUS_SCALE = 1.15
VEIN_CUT_RADIUS_PERCENTILE = 98.0
VEIN_CUT_CYLINDER_VERTICES = 64
VEIN_CUT_DEPTH_MARGIN_MM = 6.0
VEIN_CUT_MIN_RADIUS_MM = 6.0
VEIN_CUT_MIN_DEPTH_MM = 12.0
MITRAL_CUT_OFFSET_MM = 10.0

class MESH_OT_KnifeCutter(bpy.types.Operator):
    """Press K to toggle Cut Mode, S to save cuts, E to Export, ESC to cancel."""
    bl_idname = "mesh.knife_cutter"
    bl_label = "Interactive Knife Cutter"

    _timer = None
    _active_session_id = 0
    _is_cutting = False
    target = None
    cutters = []
    _export_matrix = None
    _original_verts = None   # list[Vector] – original mesh vertex positions
    _original_edges = None   # list[tuple[int,int]]
    _original_faces = None   # list[list[int]]
    _bisect_eps = None       # float – absolute epsilon for this model
    _session_id = None

    # ------------------------------------------------------------------
    # Object validity helpers
    # ------------------------------------------------------------------

    def _object_exists(self, obj):
        if obj is None:
            return False
        try:
            obj.name
            return True
        except ReferenceError:
            return False

    def _get_live_cutters(self):
        live = [c for c in self.cutters if self._object_exists(c)]
        self.cutters = live
        return live

    def _require_live_scene_objects(self):
        if not self._object_exists(self.target):
            self.report({'ERROR'}, "Target mesh is no longer valid. Rerun the script.")
            self._is_cutting = False
            self.target = None
            self.cutters = []
            return False

        if not self._get_live_cutters():
            self.report({'ERROR'}, "Cutters are no longer valid. Rerun the script.")
            self._is_cutting = False
            return False

        return True

    # ------------------------------------------------------------------
    # Path helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _blend_base_dir(self):
        blend_filepath = bpy.data.filepath
        if blend_filepath:
            return os.path.dirname(blend_filepath)
        return os.getcwd()

    def resolve_path(self, path, allow_missing_parent=False):
        """Resolve Blender-style // paths, with optional script-dir fallback."""
        path_str = str(path)
        if path_str == "//":
            resolved = self._blend_base_dir()
        elif path_str.startswith("//"):
            resolved = os.path.normpath(
                os.path.join(self._blend_base_dir(), path_str[2:])
            )
        else:
            resolved = bpy.path.abspath(path_str)

        if os.path.exists(resolved):
            return resolved

        if USE_SCRIPT_DIR_FALLBACK:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_relative = path_str[2:] if path_str.startswith("//") else path_str
            fallback = os.path.normpath(os.path.join(script_dir, script_relative))
            if allow_missing_parent:
                fallback_parent = os.path.dirname(fallback)
                if not fallback_parent or os.path.exists(fallback_parent):
                    return fallback
            elif os.path.exists(fallback):
                return fallback

        return resolved

    # ------------------------------------------------------------------
    # YAML save / load (unchanged from original)
    # ------------------------------------------------------------------

    def _format_vector_yaml(self, values):
        return "[{}]".format(", ".join(f"{float(v):.8f}" for v in values))

    def _parse_vector_yaml(self, value, field_name, cutter_name):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} for {cutter_name}: {value}") from exc
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError(f"Invalid {field_name} for {cutter_name}: {value}")
        return tuple(float(c) for c in parsed)

    def save_cut_transforms(self):
        live_cutters = self._get_live_cutters()
        if not live_cutters:
            self.report({'WARNING'}, "No cutters available to save.")
            return False

        resolved_path = self.resolve_path(CUTS_FILE_PATH, allow_missing_parent=True)
        cuts_dir = os.path.dirname(resolved_path)
        if cuts_dir and os.path.isfile(cuts_dir):
            cuts_dir = os.path.dirname(cuts_dir)
            resolved_path = os.path.join(cuts_dir, os.path.basename(resolved_path))
        if cuts_dir:
            os.makedirs(cuts_dir, exist_ok=True)

        lines = ["cutters:"]
        for cutter in live_cutters:
            lines.extend([
                f"  - name: {cutter.name}",
                f"    location: {self._format_vector_yaml(cutter.location)}",
                f"    rotation: {self._format_vector_yaml(cutter.rotation_euler)}",
                f"    scale: {self._format_vector_yaml(cutter.scale)}",
            ])

        with open(resolved_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        self.report({'INFO'}, f"Saved cut transforms to {resolved_path}")
        print(f"SAVED CUTS: {resolved_path}")
        return True

    def load_cut_transforms(self):
        if not self._require_live_scene_objects():
            return False

        resolved_path = self.resolve_path(CUTS_FILE_PATH)
        if not os.path.exists(resolved_path):
            self.report({'WARNING'}, f"Cuts file not found: {resolved_path}")
            return False

        cutter_states = []
        current_state = None

        with open(resolved_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line == "cutters:":
                    continue
                if line.startswith("- name:"):
                    if current_state is not None:
                        cutter_states.append(current_state)
                    current_state = {"name": line.split(":", 1)[1].strip()}
                    continue
                if current_state is None or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                current_state[key.strip()] = value.strip()

        if current_state is not None:
            cutter_states.append(current_state)

        cutters_by_name = {c.name: c for c in self.cutters}
        applied_count = 0

        for index, state in enumerate(cutter_states):
            cutter_name = state.get("name")
            cutter = cutters_by_name.get(cutter_name)
            if cutter is None:
                if index >= len(self.cutters):
                    continue
                cutter = self.cutters[index]
                print(f"LOAD CUTS: Matched {cutter_name} to {cutter.name} by order.")

            try:
                cutter.location = self._parse_vector_yaml(
                    state["location"], "location", cutter_name
                )
                cutter.rotation_mode = 'XYZ'
                cutter.rotation_euler = self._parse_vector_yaml(
                    state["rotation"], "rotation", cutter_name
                )
                cutter.scale = self._parse_vector_yaml(
                    state["scale"], "scale", cutter_name
                )
            except KeyError as exc:
                self.report({'ERROR'}, f"Missing {exc.args[0]} for {cutter_name} in {resolved_path}")
                return False
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return False

            applied_count += 1

        bpy.context.view_layer.update()
        self.report({'INFO'}, f"Loaded {applied_count} cutter transforms from {resolved_path}")
        print(f"LOADED CUTS: {resolved_path}")
        return True

    # ------------------------------------------------------------------
    # Cutter placement helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _build_spawn_directions(self, count):
        base_dirs = [
            Vector((3/4, 1/8, -1/8)), Vector((-3/4, 1/8, -1/8)),
            Vector((3/4, 1/8, 1/2)), Vector((-3/4, 1/8, 1/2)),
            Vector((1/4, -1/4, 1/2)), Vector((0, 0, -1)),
            Vector((1, 1, 0)).normalized(), Vector((-1, 1, 0)).normalized(),
            Vector((1, 0, 1)).normalized(), Vector((-1, 0, 1)).normalized(),
            Vector((0, 1, 1)).normalized(), Vector((0, -1, 1)).normalized(),
        ]
        return [base_dirs[i % len(base_dirs)] for i in range(count)]

    def _outside_location_from_direction(self, direction, dim, clearance):
        return (
            direction.x * (dim.x * 0.5 + clearance),
            direction.y * (dim.y * 0.5 + clearance),
            direction.z * (dim.z * 0.5 + clearance),
        )

    def _create_cutters(self, context, dim, avg):
        self.cutters = []
        if AUTO_CREATE_VEIN_CUTTERS:
            self._create_vein_cutters(context)

        total = N_EXTRA_CUT_CUBES + N_EXTRA_CUT_CYLINDERS
        if total <= 0:
            return

        cube_scale = avg * CUBE_SCALE_FACTOR
        cyl_radius = avg * CYLINDER_RADIUS_FACTOR
        cyl_depth = avg * CYLINDER_DEPTH_FACTOR
        max_extent = max(cube_scale * 0.5, cyl_radius, cyl_depth * 0.5)
        clearance = avg * SPAWN_MARGIN_FACTOR + max_extent
        directions = self._build_spawn_directions(total)

        for i in range(N_EXTRA_CUT_CUBES):
            pos = self._outside_location_from_direction(directions[i], dim, clearance)
            bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
            c = context.active_object
            c.name = f"Cut_Cube_Extra_{i + 1}"
            c.scale = (cube_scale, cube_scale, cube_scale)
            c.display_type = 'WIRE'
            c.parent = self.target
            self.cutters.append(c)

        for i in range(N_EXTRA_CUT_CYLINDERS):
            idx = N_EXTRA_CUT_CUBES + i
            pos = self._outside_location_from_direction(directions[idx], dim, clearance)
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1, location=pos)
            c = context.active_object
            c.name = f"Cut_Cylinder_Extra_{i + 1}"
            if i == 0:
                c.location = EXTRA_CYLINDER_DEFAULT_LOCATION
                c.rotation_mode = 'XYZ'
                c.rotation_euler = tuple(radians(angle) for angle in EXTRA_CYLINDER_DEFAULT_ROTATION_DEG)
                c.scale = EXTRA_CYLINDER_DEFAULT_SCALE
            else:
                c.scale = (cyl_radius, cyl_radius, cyl_depth)
            c.display_type = 'WIRE'
            c.parent = self.target
            self.cutters.append(c)

    def _build_neighbor_vertices_ids(self):
        neighbor_sets = [set() for _ in self.target.data.vertices]
        for edge in self.target.data.edges:
            v0, v1 = edge.vertices
            neighbor_sets[v0].add(v1)
            neighbor_sets[v1].add(v0)

        return [np.asarray(sorted(neighbors), dtype=int) for neighbors in neighbor_sets]

    def _cluster_tip_candidate_ids(self, candidate_ids, vertex_to_com_distance, vertices):
        if candidate_ids.size == 0:
            return np.asarray([], dtype=int)

        parents = list(range(len(candidate_ids)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        candidate_vertices = vertices[candidate_ids]
        for left in range(len(candidate_ids)):
            deltas = candidate_vertices[left + 1:] - candidate_vertices[left]
            close_offsets = np.where(np.linalg.norm(deltas, axis=1) <= TIP_CLUSTER_DISTANCE)[0]
            for offset in close_offsets:
                union(left, left + 1 + int(offset))

        clusters = {}
        for local_index, candidate_id in enumerate(candidate_ids):
            clusters.setdefault(find(local_index), []).append(int(candidate_id))

        tip_ids = []
        for cluster_member_ids in clusters.values():
            cluster_member_ids = np.asarray(cluster_member_ids, dtype=int)
            tip_ids.append(int(cluster_member_ids[np.argmax(vertex_to_com_distance[cluster_member_ids])]))

        return np.asarray(tip_ids, dtype=int)

    def _identify_tip_regions(self):
        vertices = np.asarray([vert.co[:] for vert in self.target.data.vertices], dtype=float)
        if len(vertices) == 0:
            return None

        center_of_mass = np.mean(vertices, axis=0)
        vertex_to_com_distance = np.linalg.norm(vertices - center_of_mass, axis=1)
        neighbor_vertices_ids = self._build_neighbor_vertices_ids()

        highest_vertex_id_of_each_trail = np.zeros(len(vertices), dtype=int)
        for vertex_id in range(len(vertices)):
            current = vertex_id
            while True:
                neighbors = neighbor_vertices_ids[current]
                if neighbors.size == 0:
                    break

                neighbor_distances = vertex_to_com_distance[neighbors]
                next_vertex_id = int(neighbors[np.argmax(neighbor_distances)])
                if vertex_to_com_distance[next_vertex_id] > vertex_to_com_distance[current]:
                    current = next_vertex_id
                else:
                    break

            highest_vertex_id_of_each_trail[vertex_id] = current

        highest_vertex_ids = np.unique(highest_vertex_id_of_each_trail)
        tip_vertex_ids = self._cluster_tip_candidate_ids(
            highest_vertex_ids,
            vertex_to_com_distance,
            vertices,
        )
        if tip_vertex_ids.size == 0:
            return None

        vertex_labels = highest_vertex_id_of_each_trail.copy()
        for highest_vertex_id in highest_vertex_ids:
            cluster_tip_candidates = tip_vertex_ids[
                np.linalg.norm(vertices[tip_vertex_ids] - vertices[highest_vertex_id], axis=1) <= TIP_CLUSTER_DISTANCE
            ]
            if cluster_tip_candidates.size == 0:
                continue

            candidate_distances = vertex_to_com_distance[cluster_tip_candidates]
            tip_vertex_id = int(cluster_tip_candidates[np.argmax(candidate_distances)])
            vertex_labels[vertex_labels == highest_vertex_id] = tip_vertex_id

        distance_order = np.argsort(vertex_to_com_distance[tip_vertex_ids])
        pulmonary_tip_count = min(N_PULMONARY_VEIN_TIPS, len(tip_vertex_ids))
        pulmonary_tip_vertex_ids = tip_vertex_ids[distance_order[-pulmonary_tip_count:]]

        mitral_tip_vertex_id = None
        other_tip_vertex_ids = tip_vertex_ids[distance_order[:-pulmonary_tip_count]]
        if other_tip_vertex_ids.size > 0:
            other_region_sizes = [
                np.sum(vertex_labels == tip_vertex_id)
                for tip_vertex_id in other_tip_vertex_ids
            ]
            mitral_tip_vertex_id = int(other_tip_vertex_ids[np.argmax(other_region_sizes)])

        return {
            'center_of_mass': center_of_mass,
            'vertices': vertices,
            'vertex_to_com_distance': vertex_to_com_distance,
            'vertex_labels': vertex_labels,
            'tip_vertex_ids': pulmonary_tip_vertex_ids,
            'pulmonary_tip_vertex_ids': pulmonary_tip_vertex_ids,
            'mitral_tip_vertex_id': mitral_tip_vertex_id,
        }

    def _estimate_vein_cutter_dimensions(self, tip_vertex_id, tip_region):
        vertices = tip_region['vertices']
        center_of_mass = tip_region['center_of_mass']
        vertex_labels = tip_region['vertex_labels']

        tip_vertex = vertices[tip_vertex_id]
        inward_direction = center_of_mass - tip_vertex
        inward_norm = np.linalg.norm(inward_direction)
        if inward_norm == 0:
            return None

        inward_direction /= inward_norm
        plane_point = tip_vertex + inward_direction * VEIN_CUT_OFFSET_MM

        region_vertices = vertices[vertex_labels == tip_vertex_id]
        if len(region_vertices) == 0:
            return None

        cut_center = plane_point
        ring_radius = self._find_cut_ring_radius(plane_point, inward_direction)

        relative = region_vertices - plane_point
        signed_distances = relative @ inward_direction

        if ring_radius is not None:
            cut_center = ring_radius['center']
            radius = ring_radius['radius'] * VEIN_CUT_RADIUS_SCALE + VEIN_CUT_RADIUS_MARGIN_MM
        else:
            plane_band = np.abs(signed_distances) <= VEIN_CUT_PLANE_WINDOW_MM
            band_vertices = region_vertices[plane_band]
            if len(band_vertices) < 3:
                band_vertices = region_vertices[signed_distances <= VEIN_CUT_PLANE_WINDOW_MM]
            if len(band_vertices) == 0:
                band_vertices = region_vertices

            band_relative = band_vertices - plane_point
            band_parallel = np.outer(band_relative @ inward_direction, inward_direction)
            band_perpendicular = band_relative - band_parallel
            band_radius = np.max(np.linalg.norm(band_perpendicular, axis=1))

            outward_region_vertices = region_vertices[signed_distances <= VEIN_CUT_PLANE_WINDOW_MM]
            if len(outward_region_vertices) == 0:
                outward_region_vertices = region_vertices

            outward_relative = outward_region_vertices - plane_point
            outward_parallel = np.outer(outward_relative @ inward_direction, inward_direction)
            outward_perpendicular = outward_relative - outward_parallel
            outward_radii = np.linalg.norm(outward_perpendicular, axis=1)
            region_radius = np.percentile(outward_radii, VEIN_CUT_RADIUS_PERCENTILE)

            radius = max(band_radius, region_radius) * VEIN_CUT_RADIUS_SCALE + VEIN_CUT_RADIUS_MARGIN_MM
        radius = max(radius, VEIN_CUT_MIN_RADIUS_MM)

        signed_distances_from_cut_center = (region_vertices - cut_center) @ inward_direction
        outward_distances = -signed_distances_from_cut_center[signed_distances_from_cut_center < 0]
        if outward_distances.size == 0:
            outward_depth = VEIN_CUT_MIN_DEPTH_MM
        else:
            outward_depth = np.max(outward_distances) + VEIN_CUT_DEPTH_MARGIN_MM
            outward_depth = max(outward_depth, VEIN_CUT_MIN_DEPTH_MM)

        return {
            'plane_point': cut_center,
            'inward_direction': inward_direction,
            'radius': float(radius),
            'depth': float(outward_depth),
        }

    def _estimate_mitral_cutter_dimensions(self, tip_region):
        mitral_tip_vertex_id = tip_region['mitral_tip_vertex_id']
        if mitral_tip_vertex_id is None:
            return None

        vertices = tip_region['vertices']
        center_of_mass = tip_region['center_of_mass']
        vertex_labels = tip_region['vertex_labels']

        tip_vertex = vertices[mitral_tip_vertex_id]
        outward_direction = tip_vertex - center_of_mass
        outward_norm = np.linalg.norm(outward_direction)
        if outward_norm == 0:
            return None

        outward_direction /= outward_norm
        inward_direction = -outward_direction
        plane_point = tip_vertex + inward_direction * MITRAL_CUT_OFFSET_MM

        region_vertices = vertices[vertex_labels == mitral_tip_vertex_id]
        if len(region_vertices) == 0:
            return None

        return {
            'plane_point': plane_point,
            'outward_direction': outward_direction,
            'radius': float(EXTRA_CYLINDER_DEFAULT_SCALE[0]),
            'depth': float(EXTRA_CYLINDER_DEFAULT_SCALE[2]),
        }

    def _get_plane_intersection_point_id(self, point_by_key, points, parents, key, point):
        if key in point_by_key:
            return point_by_key[key]

        point_id = len(points)
        point_by_key[key] = point_id
        points.append(point)
        parents.append(point_id)
        return point_id

    def _find_parent(self, parents, index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union_parents(self, parents, left, right):
        left_root = self._find_parent(parents, left)
        right_root = self._find_parent(parents, right)
        if left_root != right_root:
            parents[right_root] = left_root

    def _find_cut_ring_radius(self, plane_point, plane_normal):
        mesh = self.target.data
        vertices = np.asarray([vert.co[:] for vert in mesh.vertices], dtype=float)
        signed_distances = (vertices - plane_point) @ plane_normal
        eps = max(self._bisect_eps or 0.0, 1e-8)

        point_by_key = {}
        points = []
        parents = []

        for poly in mesh.polygons:
            poly_vertex_ids = list(poly.vertices)
            poly_point_ids = []
            for index, v0 in enumerate(poly_vertex_ids):
                v1 = poly_vertex_ids[(index + 1) % len(poly_vertex_ids)]
                d0 = signed_distances[v0]
                d1 = signed_distances[v1]

                if abs(d0) <= eps:
                    point_id = self._get_plane_intersection_point_id(
                        point_by_key, points, parents, ('v', int(v0)), vertices[v0]
                    )
                    poly_point_ids.append(point_id)
                    continue

                if abs(d1) <= eps:
                    point_id = self._get_plane_intersection_point_id(
                        point_by_key, points, parents, ('v', int(v1)), vertices[v1]
                    )
                    poly_point_ids.append(point_id)
                    continue

                if d0 * d1 > 0:
                    continue

                t = d0 / (d0 - d1)
                point = vertices[v0] + t * (vertices[v1] - vertices[v0])
                edge_key = ('e', min(int(v0), int(v1)), max(int(v0), int(v1)))
                point_id = self._get_plane_intersection_point_id(
                    point_by_key, points, parents, edge_key, point
                )
                poly_point_ids.append(point_id)

            poly_point_ids = list(dict.fromkeys(poly_point_ids))
            if len(poly_point_ids) < 2:
                continue

            first_point_id = poly_point_ids[0]
            for point_id in poly_point_ids[1:]:
                self._union_parents(parents, first_point_id, point_id)

        if not points:
            return None

        components = {}
        for point_id, point in enumerate(points):
            root = self._find_parent(parents, point_id)
            components.setdefault(root, []).append(point)

        best_points = None
        best_distance = None
        for component_points in components.values():
            if len(component_points) < 3:
                continue

            component_points = np.asarray(component_points, dtype=float)
            closest_distance = np.min(np.linalg.norm(component_points - plane_point, axis=1))
            if best_distance is None or closest_distance < best_distance:
                best_distance = closest_distance
                best_points = component_points

        if best_points is None:
            return None

        center = np.mean(best_points, axis=0)
        relative = best_points - center
        parallel = np.outer(relative @ plane_normal, plane_normal)
        perpendicular = relative - parallel
        radius = np.max(np.linalg.norm(perpendicular, axis=1))
        return {'center': center, 'radius': float(radius)}

    def _create_vein_cutters(self, context):
        tip_region = self._identify_tip_regions()
        if tip_region is None:
            return

        for index, tip_vertex_id in enumerate(tip_region['pulmonary_tip_vertex_ids'], start=1):
            cutter_dims = self._estimate_vein_cutter_dimensions(int(tip_vertex_id), tip_region)
            if cutter_dims is None:
                continue

            center = cutter_dims['plane_point'] - cutter_dims['inward_direction'] * (cutter_dims['depth'] * 0.5)
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=VEIN_CUT_CYLINDER_VERTICES,
                radius=1,
                depth=1,
                location=tuple(center),
            )
            cutter = context.active_object
            cutter.name = f"Cut_Vein_{index}"
            cutter.scale = (
                cutter_dims['radius'],
                cutter_dims['radius'],
                cutter_dims['depth'],
            )
            cutter.rotation_mode = 'QUATERNION'
            cutter.rotation_quaternion = Vector(cutter_dims['inward_direction']).to_track_quat('Z', 'Y')
            cutter.rotation_mode = 'XYZ'
            cutter.display_type = 'WIRE'
            cutter.parent = self.target
            self.cutters.append(cutter)

        mitral_dims = self._estimate_mitral_cutter_dimensions(tip_region)
        if mitral_dims is None:
            return

        center = mitral_dims['plane_point'] + mitral_dims['outward_direction'] * (mitral_dims['depth'] * 0.5)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=VEIN_CUT_CYLINDER_VERTICES,
            radius=1,
            depth=1,
            location=tuple(center),
        )
        cutter = context.active_object
        cutter.name = "Cut_Mitral_Valve"
        cutter.scale = (
            mitral_dims['radius'],
            mitral_dims['radius'],
            mitral_dims['depth'],
        )
        cutter.rotation_mode = 'QUATERNION'
        cutter.rotation_quaternion = Vector(mitral_dims['outward_direction']).to_track_quat('Z', 'Y')
        cutter.rotation_mode = 'XYZ'
        cutter.display_type = 'WIRE'
        cutter.parent = self.target
        self.cutters.append(cutter)

    def _ensure_tip_marker_material(self, name, color):
        material = bpy.data.materials.get(name)
        if material is None:
            material = bpy.data.materials.new(name=name)

        material.diffuse_color = color
        return material

    def _create_tip_vertex_markers(self, context, avg):
        if not SHOW_TIP_VERTEX_MARKERS:
            return

        tip_region = self._identify_tip_regions()
        if tip_region is None:
            return

        pulmonary_marker_material = self._ensure_tip_marker_material(
            "PulmonaryVeinTipMarker",
            (1.0, 0.0, 0.0, 1.0),
        )
        mitral_marker_material = self._ensure_tip_marker_material(
            "MitralValveTipMarker",
            (0.0, 0.25, 1.0, 1.0),
        )
        marker_radius = avg * TIP_MARKER_RADIUS_FACTOR

        for index, vertex_index in enumerate(tip_region['pulmonary_tip_vertex_ids'], start=1):
            marker_location = self.target.matrix_world @ self.target.data.vertices[int(vertex_index)].co

            bpy.ops.mesh.primitive_uv_sphere_add(radius=marker_radius, location=marker_location)
            marker = context.active_object
            marker.name = f"Pulmonary_Vein_Tip_{index}"
            marker.data.materials.clear()
            marker.data.materials.append(pulmonary_marker_material)

        mitral_tip_vertex_id = tip_region['mitral_tip_vertex_id']
        if mitral_tip_vertex_id is None:
            return

        marker_location = self.target.matrix_world @ self.target.data.vertices[int(mitral_tip_vertex_id)].co
        bpy.ops.mesh.primitive_uv_sphere_add(radius=marker_radius, location=marker_location)
        marker = context.active_object
        marker.name = "Mitral_Valve_Tip"
        marker.data.materials.clear()
        marker.data.materials.append(mitral_marker_material)

    # ------------------------------------------------------------------
    # Original mesh snapshot
    # ------------------------------------------------------------------

    def _store_original_mesh(self):
        """Snapshot the target mesh topology so cuts can be undone at any time.

        NOTE: Only raw geometry (verts/edges/faces) is stored.  UV maps, sharp
        flags, and other data layers from the imported OBJ are discarded on the
        first toggle.  This is acceptable for simulation-mesh workflows where the
        cut OBJ is the final product.
        """
        mesh = self.target.data
        self._original_verts = [v.co.copy() for v in mesh.vertices]
        self._original_edges = [(e.vertices[0], e.vertices[1]) for e in mesh.edges]
        self._original_faces = [list(p.vertices) for p in mesh.polygons]

    def _restore_original_mesh(self):
        """Overwrite the target mesh data with the stored snapshot."""
        mesh = self.target.data
        mesh.clear_geometry()
        mesh.from_pydata(self._original_verts, self._original_edges, self._original_faces)
        mesh.update()

    # ------------------------------------------------------------------
    # Knife / plane-bisect cut engine
    # ------------------------------------------------------------------

    def _get_cutter_planes_in_target_local(self, cutter):
        """Return the outward face planes of *cutter* transformed into the
        target object's local coordinate space.

        Each cutter face contributes one (plane_co, plane_no) pair.  Because
        planes are derived directly from the cutter's mesh data rather than from
        a hardcoded primitive description, this works for any convex shape.

        Normal transform uses the inverse-transpose of the combined linear map
        so non-uniform cutter scales are handled correctly.
        """
        combined = self.target.matrix_world.inverted() @ cutter.matrix_world
        normal_mat = combined.to_3x3().inverted().transposed()

        planes = []
        mesh_data = cutter.data
        for poly in mesh_data.polygons:
            vert_co = mesh_data.vertices[poly.vertices[0]].co
            plane_co = combined @ vert_co
            plane_no = (normal_mat @ poly.normal).normalized()
            planes.append((plane_co, plane_no))

        return planes

    def _is_inside_planes(self, point, planes):
        """Return True if *point* is strictly inside (or on) all half-spaces.

        A point is "inside" when it is on the negative side of every outward
        plane normal, i.e. the signed distance to each plane is ≤ epsilon.
        """
        eps = self._bisect_eps
        for plane_co, plane_no in planes:
            if (point - plane_co).dot(plane_no) > eps:
                return False
        return True

    def _apply_all_cuts_to_bm(self, bm):
        """Apply every live cutter to the given bmesh in place.

        Algorithm per cutter
        --------------------
        1. Bisect the mesh along each cutter face plane (no geometry removed
           yet).  This inserts new vertices/edges exactly where the mesh
           surface crosses each cutter boundary, so every face after this step
           lies entirely on one side of every plane.  The new edges created by
           each bisect are tracked.
        2. Delete faces whose centroids are inside the cutter's convex hull.
        3. Dissolve bisect-created edges that are still manifold (2 face
           neighbors) after the deletion step.  These are interior seam lines
           on kept faces — visible as spurious wireframe lines — that carry no
           geometric information and should be merged away.  Edges with only 1
           face neighbor are the actual hole boundary and are preserved.
        4. Remove isolated vertices left behind after face deletion.
        """
        dist = self._bisect_eps

        for cutter in self._get_live_cutters():
            planes = self._get_cutter_planes_in_target_local(cutter)
            if not planes:
                continue

            # Step 1 – bisect along each plane, collecting newly created edges.
            bisect_new_edges = set()
            for plane_co, plane_no in planes:
                geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
                result = bmesh.ops.bisect_plane(
                    bm,
                    geom=geom,
                    plane_co=plane_co,
                    plane_no=plane_no,
                    dist=dist,
                    clear_inner=False,
                    clear_outer=False,
                )
                bisect_new_edges.update(
                    e for e in result.get("geom_cut", [])
                    if isinstance(e, bmesh.types.BMEdge)
                )
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

            # Step 2 – delete faces whose centroids lie inside this cutter.
            faces_to_delete = [
                f for f in bm.faces
                if self._is_inside_planes(f.calc_center_median(), planes)
            ]

            if faces_to_delete:
                bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

            # Step 3 – dissolve bisect-created edges that are interior to kept
            # faces (2 face links = seam line, not hole boundary).
            seam_edges = [
                e for e in bisect_new_edges
                if e.is_valid and len(e.link_faces) == 2
            ]
            if seam_edges:
                bmesh.ops.dissolve_edges(bm, edges=seam_edges, use_verts=True)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

            # Step 4 – remove vertices that are no longer part of any face.
            isolated_verts = [v for v in bm.verts if not v.link_faces]
            if isolated_verts:
                bmesh.ops.delete(bm, geom=isolated_verts, context='VERTS')

        bm.normal_update()

    def _prepare_export_bmesh(self, bm):
        """Finalize the export mesh for the simulation OBJ pipeline.

        The downstream loader expects triangle-only ``f`` records and ignores
        materials and normals, so we triangulate here before writing the OBJ.
        """
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.normal_update()

    def _sanitize_exported_obj(self, export_path):
        """Strip OBJ records that the simulation mesh loader does not use."""
        with open(export_path, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()

        sanitized_lines = []
        for line in lines:
            if line.startswith(('mtllib ', 'usemtl ', 'vn ', 's ')):
                continue

            if line.startswith('f '):
                tokens = line.split()[1:]
                vertex_indices = [token.split('/')[0] for token in tokens]
                sanitized_lines.append(f"f {' '.join(vertex_indices)}\n")
                continue

            sanitized_lines.append(line)

        with open(export_path, 'w', encoding='utf-8') as fid:
            fid.writelines(sanitized_lines)

        material_path = os.path.splitext(export_path)[0] + '.mtl'
        if os.path.exists(material_path):
            os.remove(material_path)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def setup_scene(self, context):
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        resolved_path = self.resolve_path(FILE_PATH)
        if not os.path.exists(resolved_path):
            self.report({'ERROR'}, f"File not found: {resolved_path}")
            return False

        bpy.ops.wm.obj_import(filepath=resolved_path)
        if not context.selected_objects:
            return False

        self.target = context.selected_objects[0]
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        self._export_matrix = self.target.matrix_world.copy()
        self.target.location = (0, 0, 0)

        # Store the uncut mesh so we can restore it on demand.
        self._store_original_mesh()

        dim = self.target.dimensions
        avg = (dim.x + dim.y + dim.z) / 3
        self._bisect_eps = avg * BISECT_EPSILON_FACTOR

        self._create_cutters(context, dim, avg)
        self._create_tip_vertex_markers(context, avg)
        if LOAD_CUTS and os.path.exists(self.resolve_path(CUTS_FILE_PATH)):
            self.load_cut_transforms()

        context.view_layer.objects.active = self.target

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    if area.spaces.active:
                        area.spaces.active.show_gizmo = True
                        area.spaces.active.show_gizmo_object_translate = True
                        area.spaces.active.show_gizmo_object_rotate = True
                        area.spaces.active.show_gizmo_object_scale = True

                    for region in area.regions:
                        if region.type == 'WINDOW':
                            with context.temp_override(window=window, area=area, region=region):
                                bpy.ops.view3d.view_all(center=False)
                            break

        return True

    # ------------------------------------------------------------------
    # Cut toggle
    # ------------------------------------------------------------------

    def toggle_cuts(self):
        if not self._require_live_scene_objects():
            return False

        if not self._is_cutting:
            # Restore the original mesh, then apply all cuts.
            self._restore_original_mesh()

            bm = bmesh.new()
            bm.from_mesh(self.target.data)
            self._apply_all_cuts_to_bm(bm)
            bm.to_mesh(self.target.data)
            bm.free()
            self.target.data.update()

            self._is_cutting = True
            print("MODE: CUTTING (Knife)")
        else:
            self._restore_original_mesh()
            self._is_cutting = False
            print("MODE: MOVE (Zero Lag)")

        return True

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _build_export_object(self, context):
        """Create a temporary export object with cuts applied at the original
        world position.  Caller is responsible for removing the object.

        A temporary copy is used so the live target (and its child cutters) are
        never moved, avoiding modal-state corruption.
        """
        # Build a fresh bmesh from the stored uncut snapshot.
        bm = bmesh.new()
        temp_mesh = bpy.data.meshes.new("_knife_export_temp")
        temp_mesh.from_pydata(
            self._original_verts, self._original_edges, self._original_faces
        )
        temp_mesh.update()
        bm.from_mesh(temp_mesh)

        self._apply_all_cuts_to_bm(bm)
        self._prepare_export_bmesh(bm)

        bm.to_mesh(temp_mesh)
        bm.free()
        temp_mesh.update()

        temp_obj = bpy.data.objects.new("_knife_export_temp", temp_mesh)
        # Apply the original world matrix so the exported vertices match the
        # position of the source OBJ file.
        temp_obj.matrix_world = self._export_matrix
        context.collection.objects.link(temp_obj)
        return temp_obj

    def export_mesh(self, context):
        """Export the cut result as a single-layer surface OBJ."""
        if not self._require_live_scene_objects():
            return False

        self.save_cut_transforms()

        resolved_export_path = self.resolve_path(EXPORT_FILE_PATH, allow_missing_parent=True)
        export_dir = os.path.dirname(resolved_export_path)
        if export_dir and os.path.isfile(export_dir):
            export_dir = os.path.dirname(export_dir)
            resolved_export_path = os.path.join(export_dir, os.path.basename(resolved_export_path))
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        print(f"Attempting to export cut mesh to: {resolved_export_path}")

        export_obj = self._build_export_object(context)
        if export_obj is None:
            return False

        success = False
        try:
            bpy.ops.object.select_all(action='DESELECT')
            export_obj.select_set(True)
            context.view_layer.objects.active = export_obj

            try:
                # Blender 3.2+
                bpy.ops.wm.obj_export(
                    filepath=resolved_export_path,
                    export_selected_objects=True,
                )
            except AttributeError:
                # Older Blender
                bpy.ops.export_scene.obj(
                    filepath=resolved_export_path,
                    use_selection=True,
                    use_mesh_modifiers=False,
                )

            self._sanitize_exported_obj(resolved_export_path)

            self.report({'INFO'}, f"Exported successfully to {resolved_export_path}")
            print(f"SUCCESS: Exported to {resolved_export_path}")
            success = True
        except Exception as exc:
            self.report({'ERROR'}, f"Export failed: {exc}")
            print(f"ERROR: Export failed: {exc}")
        finally:
            temp_mesh = export_obj.data
            bpy.data.objects.remove(export_obj, do_unlink=True)
            bpy.data.meshes.remove(temp_mesh)

            # Restore selection so the user can keep working immediately.
            context.view_layer.objects.active = self.target
            for c in self._get_live_cutters():
                c.select_set(True)

        return success

    # ------------------------------------------------------------------
    # Modal loop
    # ------------------------------------------------------------------

    def modal(self, context, event):
        if self._session_id != type(self)._active_session_id:
            return {'CANCELLED'}

        if event.type == 'K' and event.value == 'PRESS':
            if not self.toggle_cuts():
                return {'CANCELLED'}

        elif event.type == 'S' and event.value == 'PRESS':
            if not self.save_cut_transforms():
                return {'CANCELLED'}

        elif event.type == 'E' and event.value == 'PRESS':
            if not self.export_mesh(context):
                return {'CANCELLED'}

        elif event.type == 'ESC':
            print("Knife Cutter Stopped.")
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self.setup_scene(context):
            type(self)._active_session_id += 1
            self._session_id = type(self)._active_session_id
            context.window_manager.modal_handler_add(self)
            print("RUNNING: Press 'K' to toggle cuts, 'S' to save cuts, 'E' to export, 'ESC' to stop.")
            return {'RUNNING_MODAL'}
        return {'CANCELLED'}


# --- RUN SCRIPT ---
def register():
    bpy.utils.register_class(MESH_OT_KnifeCutter)


def unregister():
    bpy.utils.unregister_class(MESH_OT_KnifeCutter)


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass

    register()
    bpy.ops.mesh.knife_cutter()
