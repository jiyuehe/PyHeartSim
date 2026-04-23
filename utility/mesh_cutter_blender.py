import bpy
import os
from mathutils import Vector

# --- CONFIGURATION ---
name_prefix = '0_1-la1 78 240'

FILE_PATH ="/home/j/Desktop/hdd/share_folder/patient_data" 
#FILE_PATH = "//mesh_example/before_hole_cut.obj"

EXPORT_FILE_PATH = FILE_PATH / f"{name_prefix}_refined_cut.obj"
USE_SCRIPT_DIR_FALLBACK = True

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
    _is_cutting = False
    target = None
    cutters = []

    def resolve_mesh_path(self, path):
        """Resolve Blender-style // paths, with optional script-dir fallback."""
        resolved = bpy.path.abspath(path)
        if os.path.exists(resolved):
            return resolved

        if USE_SCRIPT_DIR_FALLBACK:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_relative = path[2:] if path.startswith("//") else path
            fallback = os.path.normpath(os.path.join(script_dir, script_relative))
            if os.path.exists(os.path.dirname(fallback)): # Only check if directory exists for exporting
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
        resolved_path = self.resolve_mesh_path(FILE_PATH)
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

    def export_mesh(self, context):
        """Exports the target mesh with all current boolean cuts applied."""
        # 1. Temporarily enforce CUTTING mode so the export has the holes
        was_cutting = self._is_cutting
        if not was_cutting:
            self.toggle_booleans()
            
        # 2. Isolate selection to the target object only
        bpy.ops.object.select_all(action='DESELECT')
        self.target.select_set(True)
        context.view_layer.objects.active = self.target
        
        # 3. Resolve path and ensure directories exist
        resolved_export_path = bpy.path.abspath(EXPORT_FILE_PATH)
        export_dir = os.path.dirname(resolved_export_path)
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

    def modal(self, context, event):
        if event.type == 'K' and event.value == 'PRESS':
            self.toggle_booleans()
            
        elif event.type == 'E' and event.value == 'PRESS':
            self.export_mesh(context)
            
        elif event.type == 'ESC':
            print("Interactive Cutter Stopped.")
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self.setup_scene(context):
            context.window_manager.modal_handler_add(self)
            print("RUNNING: Press 'K' to toggle cuts, 'E' to export, 'ESC' to stop script.")
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