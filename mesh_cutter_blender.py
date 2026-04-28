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

import bpy
import os
import ast
from mathutils import Vector
from pathlib import Path

# --- CONFIGURATION ---
name_prefix = '2_2-lafam pr'

BASE_PATH = Path("//")
#BASE_PATH = Path("/home/mason/Code/PyHeartSim/")

FILE_PATH = BASE_PATH / f"{name_prefix}_refined.obj"
EXPORT_FILE_PATH = BASE_PATH / f"{name_prefix}_refined_cut.obj"
CUTS_FILE_PATH = BASE_PATH / f"{name_prefix}_cuts.yaml"
USE_SCRIPT_DIR_FALLBACK = True
LOAD_CUTS = False

# Number of interactive cutters to create.
N_CUT_CUBES = 4
N_CUT_CYLINDERS = 1

# Cutter sizing, expressed as a fraction of average target dimension.
CUBE_SCALE_FACTOR = 0.15
CYLINDER_RADIUS_FACTOR = 0.12
CYLINDER_DEPTH_FACTOR = 0.30

# Additional spacing so cutters spawn outside the target mesh bounds.
SPAWN_MARGIN_FACTOR = 0.20

class MESH_OT_InteractiveCutter(bpy.types.Operator):
    """Press K to toggle Cut Mode, E to Export, ESC to cancel"""
    bl_idname = "mesh.interactive_cutter"
    bl_label = "Interactive Mesh Cutter"
    
    _timer = None
    _active_session_id = 0
    _is_cutting = False
    target = None
    cutters = []
    _session_id = None

    def _object_exists(self, obj):
        if obj is None:
            return False

        try:
            obj.name
            return True
        except ReferenceError:
            return False

    def _get_live_cutters(self):
        live_cutters = []
        for cutter in self.cutters:
            if self._object_exists(cutter):
                live_cutters.append(cutter)
        self.cutters = live_cutters
        return live_cutters

    def _require_live_scene_objects(self):
        if not self._object_exists(self.target):
            self.report({'ERROR'}, "Target mesh is no longer valid. Rerun the script to rebuild the scene.")
            self._is_cutting = False
            self.target = None
            self.cutters = []
            return False

        live_cutters = self._get_live_cutters()
        if not live_cutters:
            self.report({'ERROR'}, "Cutters are no longer valid. Rerun the script to rebuild the scene.")
            self._is_cutting = False
            return False

        return True

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

    def _format_vector_yaml(self, values):
        return "[{}]".format(", ".join(f"{float(value):.8f}" for value in values))

    def _parse_vector_yaml(self, value, field_name, cutter_name):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} for {cutter_name}: {value}") from exc

        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError(f"Invalid {field_name} for {cutter_name}: {value}")

        return tuple(float(component) for component in parsed)

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

        cutters_by_name = {cutter.name: cutter for cutter in self.cutters}
        applied_count = 0

        for cutter_state in cutter_states:
            cutter_name = cutter_state.get("name")
            cutter = cutters_by_name.get(cutter_name)
            if cutter is None:
                continue

            try:
                cutter.location = self._parse_vector_yaml(
                    cutter_state["location"], "location", cutter_name
                )
                cutter.rotation_mode = 'XYZ'
                cutter.rotation_euler = self._parse_vector_yaml(
                    cutter_state["rotation"], "rotation", cutter_name
                )
                cutter.scale = self._parse_vector_yaml(
                    cutter_state["scale"], "scale", cutter_name
                )
            except KeyError as exc:
                self.report({'ERROR'}, f"Missing {exc.args[0]} for {cutter_name} in {resolved_path}")
                return False
            except ValueError as exc:
                self.report({'ERROR'}, str(exc))
                return False

            applied_count += 1

        self.report({'INFO'}, f"Loaded {applied_count} cutter transforms from {resolved_path}")
        print(f"LOADED CUTS: {resolved_path}")
        return True

    def _build_spawn_directions(self, count):
        base_dirs = [
            Vector((3/4, 3/4, 0)), Vector((-3/4, -3/4, 0)),
            Vector((3/4, -3/4, 0)), Vector((-3/4, 3/4, 0)),
            Vector((0, 0, 1)), Vector((0, 0, -1)),
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
        total_cutters = N_CUT_CUBES + N_CUT_CYLINDERS
        if total_cutters <= 0:
            return

        cube_scale = avg * CUBE_SCALE_FACTOR
        cyl_radius = avg * CYLINDER_RADIUS_FACTOR
        cyl_depth = avg * CYLINDER_DEPTH_FACTOR

        max_cutter_extent = max(cube_scale * 0.5, cyl_radius, cyl_depth * 0.5)
        clearance = avg * SPAWN_MARGIN_FACTOR + max_cutter_extent
        directions = self._build_spawn_directions(total_cutters)

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

    def setup_scene(self, context):
        # 1. Clear scene safely
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
            
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # 2. Import the OBJ
        resolved_path = self.resolve_path(FILE_PATH)
        if not os.path.exists(resolved_path):
            self.report({'ERROR'}, f"File not found: {resolved_path}")
            return False
            
        bpy.ops.wm.obj_import(filepath=resolved_path)
        if not context.selected_objects:
            return False
            
        self.target = context.selected_objects[0]
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        self.target.location = (0, 0, 0)
        
        # --- THE FIX: MAKE IT A HOLLOW SHELL ---
        solid_mod = self.target.modifiers.new(name="Hollow_Shell", type='SOLIDIFY')
        solid_mod.thickness = 0.0001
        solid_mod.offset = 0.0
        
        # 3. Create Cutters
        dim = self.target.dimensions
        avg = (dim.x + dim.y + dim.z) / 3
        self._create_cutters(context, dim, avg)
        if LOAD_CUTS:
            self.load_cut_transforms()
            
        context.view_layer.objects.active = self.target
        
        # 4. Viewport Zoom & Gizmo Fix
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    # --- NEW: Enable transform gizmos by default ---
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

    def toggle_booleans(self):
        if not self._require_live_scene_objects():
            return False

        if not self._is_cutting:
            # Enable Booleans
            for c in self.cutters:
                mod = self.target.modifiers.new(name=c.name, type='BOOLEAN')
                mod.object = c
                mod.solver = 'EXACT'
                mod.operation = 'DIFFERENCE'
            self._is_cutting = True
            print("MODE: CUTTING (Live Calculation)")
        else:
            # Remove Booleans for fast moving
            for c in self.cutters:
                if c.name in self.target.modifiers:
                    self.target.modifiers.remove(self.target.modifiers[c.name])
            self._is_cutting = False
            print("MODE: MOVE (Zero Lag)")

        return True

    def export_mesh(self, context):
        """Exports the target mesh with all current boolean cuts applied."""
        if not self._require_live_scene_objects():
            return False

        self.save_cut_transforms()

        # 1. Temporarily enforce CUTTING mode so the export has the holes
        was_cutting = self._is_cutting
        if not was_cutting:
            self.toggle_booleans()
            
        # 2. Isolate selection to the target object only
        bpy.ops.object.select_all(action='DESELECT')
        self.target.select_set(True)
        context.view_layer.objects.active = self.target
        
        # 3. Resolve path and ensure directories exist
        resolved_export_path = self.resolve_path(EXPORT_FILE_PATH, allow_missing_parent=True)
        export_dir = os.path.dirname(resolved_export_path)
        if export_dir and os.path.isfile(export_dir):
            export_dir = os.path.dirname(export_dir)
            resolved_export_path = os.path.join(export_dir, os.path.basename(resolved_export_path))
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)
            
        print(f"Attempting to export cut mesh to: {resolved_export_path}")
        
        try:
            # Modern OBJ export (Blender 3.2+), applies modifiers automatically
            bpy.ops.wm.obj_export(
                filepath=resolved_export_path,
                export_selected_objects=True
            )
            self.report({'INFO'}, f"Exported successfully to {resolved_export_path}")
            print(f"SUCCESS: Exported to {resolved_export_path}")
        except AttributeError:
            # Fallback for older Blender versions (<3.2)
            bpy.ops.export_scene.obj(
                filepath=resolved_export_path,
                use_selection=True,
                use_mesh_modifiers=True
            )
            self.report({'INFO'}, f"Exported successfully to {resolved_export_path}")
            print(f"SUCCESS: Exported to {resolved_export_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            print(f"ERROR: Export failed: {str(e)}")
            
        # 4. Restore previous toggle state (if you were in Move mode, go back to it)
        if not was_cutting:
            self.toggle_booleans()
            
        # Reselect cutters so the user can keep working immediately
        for c in self.cutters:
            c.select_set(True)

        return True

    def modal(self, context, event):
        if self._session_id != type(self)._active_session_id:
            return {'CANCELLED'}

        if event.type == 'K' and event.value == 'PRESS':
            if not self.toggle_booleans():
                return {'CANCELLED'}

        elif event.type == 'S' and event.value == 'PRESS':
            if not self.save_cut_transforms():
                return {'CANCELLED'}
            
        elif event.type == 'E' and event.value == 'PRESS':
            if not self.export_mesh(context):
                return {'CANCELLED'}
            
        elif event.type == 'ESC':
            print("Interactive Cutter Stopped.")
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self.setup_scene(context):
            type(self)._active_session_id += 1
            self._session_id = type(self)._active_session_id
            context.window_manager.modal_handler_add(self)
            print("RUNNING: Press 'K' to toggle cuts, 'S' to save cuts, 'E' to export, 'ESC' to stop script.")
            return {'RUNNING_MODAL'}
        return {'CANCELLED'}

# --- RUN SCRIPT ---
def register():
    bpy.utils.register_class(MESH_OT_InteractiveCutter)

def unregister():
    bpy.utils.unregister_class(MESH_OT_InteractiveCutter)

if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
        
    register()
    bpy.ops.mesh.interactive_cutter()