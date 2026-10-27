"""One Cycles render exports clay, inverted percentile-normalized Mist, and normal."""
import bpy
import argparse
import json
import sys
import time
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--outdir', default='output')
p.add_argument('--name', default='demo')
p.add_argument('--res', type=int, choices=[512,768], default=512)
p.add_argument('--samples', type=int, default=64)
args = p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out = Path(args.outdir).resolve(); out.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
scene = bpy.context.scene
scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
scene.cycles.samples = args.samples; scene.cycles.seed = 42
scene.render.resolution_x = scene.render.resolution_y = args.res
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.color_depth = '8'
scene.render.filepath = str(out/f'{args.name}_clay.png')
layer = scene.view_layers[0]
layer.use_pass_mist = True; layer.use_pass_normal = True
depths=[]
depsgraph = bpy.context.evaluated_depsgraph_get()
for ob in scene.objects:
    if ob.type != 'MESH' or ob.name == 'Ground' or ob.hide_render:
        continue
    evaluated = ob.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    for vertex in mesh.vertices:
        co = evaluated.matrix_world @ vertex.co
        uv = world_to_camera_view(scene, scene.camera, co)
        if 0 <= uv.x <= 1 and 0 <= uv.y <= 1 and uv.z > 0:
            depths.append((co-scene.camera.location).length)
    evaluated.to_mesh_clear()
if len(depths) < 2:
    raise RuntimeError('No visible geometry for depth normalization')
near,far = [float(x) for x in np.percentile(depths,[2,98])]
mist=scene.world.mist_settings
mist.start=near; mist.depth=max(far-near,1e-3); mist.falloff='LINEAR'
scene.use_nodes=True
tree=scene.node_tree; tree.nodes.clear()
rl=tree.nodes.new('CompositorNodeRLayers')
inverse=tree.nodes.new('CompositorNodeMath'); inverse.operation='SUBTRACT'
inverse.inputs[0].default_value=1
tree.links.new(rl.outputs['Mist'],inverse.inputs[1])
scale=tree.nodes.new('CompositorNodeMixRGB'); scale.blend_type='MULTIPLY'
scale.inputs[0].default_value=1; scale.inputs[2].default_value=(0.5,0.5,0.5,1)
tree.links.new(rl.outputs['Normal'],scale.inputs[1])
offset=tree.nodes.new('CompositorNodeMixRGB'); offset.blend_type='ADD'
offset.inputs[0].default_value=1; offset.inputs[2].default_value=(0.5,0.5,0.5,1)
tree.links.new(scale.outputs[0],offset.inputs[1])
for kind,socket in [('depth',inverse.outputs[0]),('normal',offset.outputs[0])]:
    node=tree.nodes.new('CompositorNodeOutputFile')
    node.base_path=str(out)
    node.format.file_format='PNG'; node.format.color_mode='RGB'; node.format.color_depth='8'
    node.format.color_management='OVERRIDE'
    node.format.view_settings.view_transform='Raw'
    node.format.view_settings.look='None'
    node.format.view_settings.exposure=0
    node.format.view_settings.gamma=1
    node.file_slots[0].path=f'{args.name}_{kind}_'
    node.file_slots[0].save_as_render=True
    tree.links.new(socket,node.inputs[0])
composite=tree.nodes.new('CompositorNodeComposite')
tree.links.new(rl.outputs['Image'],composite.inputs[0])
scene.frame_set(1)
render_start=time.perf_counter()
bpy.ops.render.render(write_still=True)
end=time.perf_counter()
data={'name':args.name,'resolution':args.res,'samples':args.samples,'device':'Cycles CPU',
      'blender_version':bpy.app.version_string,'visible_vertices':len(depths),
      'map_view_transform':'Raw','clay_view_transform':scene.view_settings.view_transform,
      'mist_start':near,'mist_depth':far-near,'percentiles':[2,98],
      'render_seconds':end-render_start,'export_seconds':end-start,
      'timing_note':'clay/depth/normal share one render; export_seconds excludes Blender startup'}
(out/f'{args.name}_export.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print('EXPORT_RESULT',json.dumps(data))
