"""Reproduce the extra precision and convolution diagnostics, separate from A-E."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    base=json.loads((ROOT/'output/A_command.json').read_text(encoding='utf-8'))
    for name,conv in [('probe_vae_legacy','auto,vae=off'),('probe_all_legacy','off'),('probe_f16',None)]:
        cmd=base.copy();cmd[cmd.index('-o')+1]=str(ROOT/f'output/{name}.png')
        if conv:cmd+=['--conv-direct',conv]
        else:cmd[cmd.index('-m')+1]=str(ROOT/'sd_turbo-f16.gguf')
        (ROOT/f'output/{name}_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
        with (ROOT/f'logs/{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,'completed',flush=True)
if __name__=='__main__':main()
