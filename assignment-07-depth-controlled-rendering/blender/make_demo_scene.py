"""Run with Blender: blender -b --factory-startup -P blender/make_demo_scene.py."""
import bpy
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 64
scene.cycles.seed = 42
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.world.color = (0.12, 0.12, 0.12)
mat = bpy.data.materials.new('White matte')
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
bsdf.inputs['Roughness'].default_value = 0.65

def finish(name, smooth=False):
    ob = bpy.context.object
    ob.name = name
    ob.data.materials.append(mat)
    if smooth:
        for poly in ob.data.polygons:
            poly.use_smooth = True
    return ob

bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.67, location=(-1.15, -0.65, 0.67))
finish('Sphere', True)
bpy.ops.mesh.primitive_torus_add(major_radius=0.65, minor_radius=0.18, major_segments=64, minor_segments=24,
                               location=(-1.05, 1.0, 0.83), rotation=(math.pi/2, 0, 0))
finish('Torus', True)
bpy.ops.mesh.primitive_cube_add(size=1.25, location=(0.75, -0.65, 0.625))
cube = finish('Beveled cube')
cube.data.use_auto_smooth = True
bevel = cube.modifiers.new('Beveled edges', 'BEVEL')
bevel.width = 0.10
bevel.segments = 5
cube.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.48, depth=1.7, location=(1.05, 1.05, 0.85))
cyl = finish('Cylinder', True)
cyl.data.use_auto_smooth = True
bevel = cyl.modifiers.new('Rim bevel', 'BEVEL')
bevel.width = 0.05
bevel.segments = 3
cyl.modifiers.new('Weighted normals', 'WEIGHTED_NORMAL')
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.005))
finish('Ground')
bpy.ops.object.camera_add(location=(6, -9, 6))
camera = bpy.context.object
camera.name = 'Camera'
camera.rotation_euler = (Vector((0, 0.1, 0.65))-camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = 63
scene.camera = camera
for name, pos, power, size in [('Key',(-3,-4,7),1100,5),('Fill',(5,-1,4),700,4),('Rim',(0,5,6),1000,3)]:
    bpy.ops.object.light_add(type='AREA', location=pos)
    light = bpy.context.object
    light.name = name
    light.data.energy = power
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (Vector((0,0,0.5))-light.location).to_track_quat('-Z','Y').to_euler()
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1
scene.render.image_settings.file_format = 'PNG'
(ROOT/'assets').mkdir(exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'assets/demo_white_model.blend'))
print('SCENE_READY', bpy.app.version_string)
