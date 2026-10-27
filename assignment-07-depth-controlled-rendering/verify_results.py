"""Validate recorded parameter invariants, timing arithmetic and image channels."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'output/results.csv').open(encoding='utf-8')))
expected={'A':(512,4,1.2,.9),'B':(768,4,1.2,.9),'C':(512,8,1.2,.9),'D':(512,4,1.2,.5),'E':(512,4,1.,.9)}
assert [r['run'] for r in rows]==list('ABCDE')
for row in rows:
 name=row['run'];res=int(row['resolution'])
 assert (res,int(row['steps']),float(row['cfg']),float(row['control_strength']))==expected[name]
 assert int(row['seed'])==42 and row['sampler']=='euler' and row['scheduler']=='discrete'
 saved=json.loads((ROOT/f'output/{name}_result.json').read_text())
 for k in row:assert str(saved[k])==row[k],(name,k)
 assert abs(float(row['total_seconds'])-float(row['process_seconds'])-float(row['blender_seconds']))<1e-6
 for key in ['sampling_seconds','vae_seconds','process_seconds','blender_seconds']:assert float(row[key])>0
 with Image.open(ROOT/row['image']) as im:assert im.size==(res,res)
for res in [512,768]:
 with Image.open(ROOT/f'output/demo{res}_depth_0001.png') as im:a=np.array(im)
 assert a.shape==(res,res,3) and np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2])
 assert np.unique(a[:,:,0]).size>50
 with Image.open(ROOT/f'output/demo{res}_normal_0001.png') as im:a=np.array(im)
 assert not np.array_equal(a[:,:,0],a[:,:,1])
for name in ['probe_zero','probe_inverse']:
 with Image.open(ROOT/'output/A_photoreal.png') as im:a=np.array(im)
 with Image.open(ROOT/f'output/{name}_photoreal.png') as im:b=np.array(im)
 assert not np.array_equal(a,b)
print('PASS: A-E parameters, CSV/JSON agreement, timings, depth/normal maps and control probes')
