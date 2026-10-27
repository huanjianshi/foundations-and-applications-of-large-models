"""Reproducible Blender -> SD-Turbo Q4_0 + SD2.1 depth ControlNet pipeline."""
import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from blender_safe_launch import blender_environment

ROOT=Path(__file__).resolve().parents[1]
PROMPT='photorealistic render, studio soft lighting, PBR material, preserve original 3d geometry, sharp'
NEGATIVE='deformed geometry, perspective changed'

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='A')
    p.add_argument('--res',type=int,choices=[512,768],default=512)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--cfg',type=float,default=1.2)
    p.add_argument('--control-strength',type=float,default=0.9)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--control-image',type=Path)
    p.add_argument('--blender',default=os.environ.get('BLENDER_EXE','blender'))
    p.add_argument('--batch',action='store_true')
    p.add_argument('--probe',action='store_true',help='extra control=0 and inverted-depth evidence, separate from A-E')
    return p.parse_args()

def execute(command,log):
    t=time.perf_counter()
    env=blender_environment()
    for key in list(env):
        if key.startswith('PYTHON'):env.pop(key)
    with log.open('w',encoding='utf-8') as f:
        f.write('COMMAND '+subprocess.list2cmdline([str(v) for v in command])+'\n');f.flush()
        proc=subprocess.run([str(v) for v in command],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-t
    if proc.returncode:
        raise RuntimeError(f'Exit {proc.returncode}: {log}')
    return elapsed,log.read_text(encoding='utf-8',errors='replace')

def maps(res,blender):
    out=ROOT/'output'; meta=out/f'demo{res}_export.json'
    if meta.exists():return json.loads(meta.read_text(encoding='utf-8'))
    if not (ROOT/'assets/demo_white_model.blend').exists():
        execute([blender,'-b','--factory-startup','-P',ROOT/'blender/make_demo_scene.py'],ROOT/'logs/make_scene.log')
    elapsed,_=execute([blender,'-b',ROOT/'assets/demo_white_model.blend','-P',ROOT/'blender/export_control_maps.py',
                       '--','--outdir',out,'--name',f'demo{res}','--res',res],ROOT/f'logs/export{res}.log')
    data=json.loads(meta.read_text(encoding='utf-8'));data['blender_process_seconds']=elapsed
    meta.write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data

def run(name,res,steps,cfg,control,seed,threads,control_image=None,blender=None):
    meta=maps(res,blender) if control_image is None else None
    image=(control_image or ROOT/f'output/demo{res}_depth_0001.png').resolve()
    if not image.exists():raise FileNotFoundError(image)
    output=ROOT/f'output/{name}_photoreal.png'
    command=[ROOT/'vendor/sdcpp/sd-cli.exe','-m',ROOT/'models/sd_turbo-q4_0.gguf',
       '--control-net',ROOT/'models/control_depth_sd21-fp16.gguf','--control-image',image,
       '--backend','cuda0','--steps',steps,'-W',res,'-H',res,'--cfg-scale',cfg,
       '--sampling-method','euler','--scheduler','discrete','--seed',seed,'--threads',threads,
       '--control-strength',control,'-p',PROMPT,'-n',NEGATIVE,'-o',output]
    elapsed,log=execute(command,ROOT/f'logs/{name}.log')
    def timing(pattern):
        match=re.search(pattern,log)
        if not match:raise RuntimeError(f'Missing timing {pattern} in {name}')
        return float(match.group(1))
    row={'run':name,'resolution':res,'steps':steps,'cfg':cfg,'control_strength':control,'seed':seed,
         'sampler':'euler','scheduler':'discrete','threads':threads,
         'sampling_seconds':timing(r'sampling completed, taking ([\d.]+)s'),
         'vae_seconds':timing(r'decode_first_stage completed, taking ([\d.]+)s'),
         'generation_seconds':timing(r'generate_image completed in ([\d.]+)s'),
         'process_seconds':elapsed,'blender_seconds':meta['blender_process_seconds'] if meta else 0,
         'total_seconds':elapsed+(meta['blender_process_seconds'] if meta else 0),
         'image':str(output.relative_to(ROOT)),
         'control_image':str(image.relative_to(ROOT)) if image.is_relative_to(ROOT) else str(image)}
    (ROOT/f'output/{name}_result.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    (ROOT/f'output/{name}_command.json').write_text(json.dumps([str(v) for v in command],indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
    return row

def main():
    args=cli()
    for d in ['output','logs','models','assets']:(ROOT/d).mkdir(exist_ok=True)
    if args.probe:
        from PIL import Image,ImageOps
        original=ROOT/'output/demo512_depth_0001.png'
        inverse=ROOT/'output/probe_inverted_depth.png'
        ImageOps.invert(Image.open(original).convert('RGB')).save(inverse)
        for name,c,image in [('probe_zero',0,original),('probe_inverse',0.9,inverse)]:
            run(name,512,4,1.2,c,42,args.threads,image,args.blender)
    elif args.batch:
        rows=[]
        for name,res,steps,cfg,control in [('A',512,4,1.2,.9),('B',768,4,1.2,.9),('C',512,8,1.2,.9),
                                            ('D',512,4,1.2,.5),('E',512,4,1.0,.9)]:
            rows.append(run(name,res,steps,cfg,control,42,args.threads,None,args.blender))
            with (ROOT/'output/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        run(args.name,args.res,args.steps,args.cfg,args.control_strength,args.seed,args.threads,args.control_image,args.blender)

if __name__=='__main__':main()
