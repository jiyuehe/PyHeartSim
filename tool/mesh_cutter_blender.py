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

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import common

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

debug_plot = 0
if debug_plot == 1:
    file_dir = Path('/home/j/Desktop/hdd/share_folder/patient_data')
    mesh_name = '99_2-LaFAM_cartofinder_data'
    vertex, face = common.load_obj(file_dir, mesh_name+'_refined')

    center_of_mass = vertex.mean(axis=0)

    # load the 4 pulmonery vein tip vertices
    tip_vertex = np.array([list(map(float, i.split())) for i in open(file_dir / f'{mesh_name}_tip_vertex.txt').read().splitlines()])

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

        # top 4 tip vertices as large red dots
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
            x=np.stack([tip_vertex[:, 0], np.full(4, center_of_mass[0]), np.full(4, None)], axis=1).ravel(),
            y=np.stack([tip_vertex[:, 1], np.full(4, center_of_mass[1]), np.full(4, None)], axis=1).ravel(),
            z=np.stack([tip_vertex[:, 2], np.full(4, center_of_mass[2]), np.full(4, None)], axis=1).ravel(),
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

# Number of interactive cutters to create.
N_CUT_CUBES = 4
N_CUT_CYLINDERS = 1

# Cutter sizing, expressed as a fraction of average target dimension.
CUBE_SCALE_FACTOR = 0.3
CYLINDER_RADIUS_FACTOR = 0.17
CYLINDER_DEPTH_FACTOR = 0.5

# Additional spacing so cutters spawn outside the target mesh bounds.
SPAWN_MARGIN_FACTOR = 0.20

# Relative epsilon for plane-distance tests and bisect snapping.  Applied as a
# fraction of the target mesh's average dimension so it scales with the model.
BISECT_EPSILON_FACTOR = 1e-5

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
        total = N_CUT_CUBES + N_CUT_CYLINDERS
        if total <= 0:
            return

        cube_scale = avg * CUBE_SCALE_FACTOR
        cyl_radius = avg * CYLINDER_RADIUS_FACTOR
        cyl_depth = avg * CYLINDER_DEPTH_FACTOR
        max_extent = max(cube_scale * 0.5, cyl_radius, cyl_depth * 0.5)
        clearance = avg * SPAWN_MARGIN_FACTOR + max_extent
        directions = self._build_spawn_directions(total)

        for i in range(N_CUT_CUBES):
            pos = self._outside_location_from_direction(directions[i], dim, clearance)
            bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
            c = context.active_object
            c.name = f"Cut_Cube_{i + 1}"
            c.scale = (cube_scale, cube_scale, cube_scale)
            c.display_type = 'WIRE'
            c.parent = self.target
            self.cutters.append(c)

        for i in range(N_CUT_CYLINDERS):
            idx = N_CUT_CUBES + i
            pos = self._outside_location_from_direction(directions[idx], dim, clearance)
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=1, location=pos)
            c = context.active_object
            c.name = f"Cut_Cylinder_{i + 1}"
            c.scale = (cyl_radius, cyl_radius, cyl_depth)
            c.display_type = 'WIRE'
            c.parent = self.target
            self.cutters.append(c)

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
