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
from mathutils import Vector

# --- CONFIGURATION ---
#FILE_PATH = "/home/j/Desktop/hdd/share_folder/patient_data/0_1-la1 78 240_refined.obj"
FILE_PATH = "/home/j/Desktop/hdd/share_folder/patient_data/0_1-la1 78 240_refined.obj"
USE_SCRIPT_DIR_FALLBACK = True
#EXPORT_PATH = "/home/j/Desktop/hdd/share_folder/patient_data/0_1-la1 78 240_refined_cut.obj"
EXPORT_PATH = "/home/j/Desktop/hdd/share_folder/patient_data/0_1-la1 78 240_refined_cut.obj"

# Number of interactive cutters to create.
N_CUT_CUBES = 3
N_CUT_CYLINDERS = 1

# Cutter sizing, expressed as a fraction of average target dimension.
CUBE_SCALE_FACTOR = 0.15
CYLINDER_RADIUS_FACTOR = 0.12
CYLINDER_DEPTH_FACTOR = 0.30

# Additional spacing so cutters spawn outside the target mesh bounds.
SPAWN_MARGIN_FACTOR = 0.20

# Viewport defaults for transform controls.
ENABLE_GIZMOS = True
DEFAULT_TRANSFORM_ORIENTATION = 'LOCAL'

class MESH_OT_InteractiveCutter(bpy.types.Operator):
    """Press K to toggle between Move (Fast) and Cut (Live) modes"""
    bl_idname = "mesh.interactive_cutter"
    bl_label = "Interactive Mesh Cutter"
    
    _timer = None
    _is_cutting = False
    target = None
    cutters = []

    def configure_viewport_defaults(self, context):
        """Enable transform gizmos in all 3D views and default to local orientation."""
        if not ENABLE_GIZMOS:
            return

        for window in context.window_manager.windows:
            screen = window.screen
            if not screen:
                continue

            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue

                space = area.spaces.active
                if not space:
                    continue

                space.show_gizmo = True
                space.show_gizmo_object_translate = True
                space.show_gizmo_object_rotate = True
                space.show_gizmo_object_scale = True
                space.transform_orientation = DEFAULT_TRANSFORM_ORIENTATION

        # Keep operator transforms on LOCAL axes too.
        context.scene.transform_orientation_slots[0].type = DEFAULT_TRANSFORM_ORIENTATION

    def export_target_mesh(self):
        if self.target is None:
            self.report({'ERROR'}, "No target mesh to export.")
            return

        export_path = bpy.path.abspath(EXPORT_PATH)
        export_dir = os.path.dirname(export_path)
        os.makedirs(export_dir, exist_ok=True)

        bpy.ops.object.select_all(action='DESELECT')
        self.target.select_set(True)
        bpy.context.view_layer.objects.active = self.target

        # Blender's OBJ exporter options differ across versions.
        try:
            bpy.ops.wm.obj_export(filepath=export_path, export_selected_objects=True)
        except TypeError:
            bpy.ops.export_scene.obj(filepath=export_path, use_selection=True)

        self.report({'INFO'}, f"Exported target mesh to: {export_path}")
        print(f"EXPORTED: {export_path}")

    def resolve_mesh_path(self):
        """Resolve Blender-style // paths, with optional script-dir fallback."""
        resolved = bpy.path.abspath(FILE_PATH)
        if os.path.exists(resolved):
            return resolved

        if USE_SCRIPT_DIR_FALLBACK:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_relative = FILE_PATH[2:] if FILE_PATH.startswith("//") else FILE_PATH
            fallback = os.path.normpath(os.path.join(script_dir, script_relative))
            if os.path.exists(fallback):
                return fallback

        return resolved

    def _build_spawn_directions(self, count):
        base_dirs = [
            Vector((1, 0, 0)), Vector((-1, 0, 0)),
            Vector((0, 1, 0)), Vector((0, -1, 0)),
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
        resolved_path = self.resolve_mesh_path()
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
        # We add a Solidify modifier to create a microscopically thin wall.
        # When the Boolean cuts through this wall, it leaves an open hole.
        solid_mod = self.target.modifiers.new(name="Hollow_Shell", type='SOLIDIFY')
        solid_mod.thickness = 0.0001  # Extremely thin
        solid_mod.offset = 0.0        # Keep it centered on original geometry
        
        # 3. Create Cutters
        dim = self.target.dimensions
        avg = (dim.x + dim.y + dim.z) / 3
        self._create_cutters(context, dim, avg)
            
        context.view_layer.objects.active = self.target
        
        # 4. Viewport Zoom Fix
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            with context.temp_override(window=window, area=area, region=region):
                                bpy.ops.view3d.view_all(center=False)
                            break
                            
        return True

    def toggle_booleans(self):
        if not self._is_cutting:
            # Enable Booleans (They will automatically stack below the Solidify modifier)
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

    def modal(self, context, event):
        if event.type == 'K' and event.value == 'PRESS':
            self.toggle_booleans()

        elif event.type == 'E' and event.value == 'PRESS':
            self.export_target_mesh()
            
        elif event.type == 'ESC':
            print("Interactive Cutter Stopped.")
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self.setup_scene(context):
            self.configure_viewport_defaults(context)
            context.window_manager.modal_handler_add(self)
            print("RUNNING: Press 'K' to toggle cuts, 'E' to export OBJ, 'ESC' to stop script.")
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