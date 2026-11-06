"""Reproduce the baseline, 16-run ablation, A/B/D/E and three paired repairs."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import distill_lm as d


def run(args):
    root=Path(__file__).resolve().parent
    out=args.output.resolve()
    subprocess.run([sys.executable,'-B',str(root/'distill_lm.py'),'--mode','complete',
                    '--device',args.device,'--output',str(out)],check=True)
    torch.set_num_threads(8)
    if args.device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    d.TOKENIZER=d.Tokenizer(d.build_corpus())
    d.TRAIN=torch.tensor(d.TOKENIZER.encode(d.build_corpus()),dtype=torch.long)
    d.VAL=torch.tensor(d.TOKENIZER.encode(d.build_corpus(300,999)),dtype=torch.long)
    teacher=d.make_model(160,0,args.device)
    teacher.load_state_dict(torch.load(out/'teacher_c160.pt',map_location=args.device,weights_only=True)['state_dict'])
    teacher_hash=d.state_hash(teacher)
    baseline=json.loads((out/'results_complete.json').read_text(encoding='utf-8'))
    repairs=[]
    checks={}
    for i,(steps,temperature) in enumerate([(150,4.),(150,8.),(180,8.)],1):
        folder=out/f'repair_round{i}'
        folder.mkdir(exist_ok=True)
        cfg=argparse.Namespace(device=args.device,seed=0,batch_size=24,lr=.003,output=folder)
        _,scratch=d.experiment(cfg,f'scratch{steps}',64,steps,reference_teacher=teacher)
        student,kd=d.experiment(cfg,f'kd{steps}',64,steps,teacher,T=temperature)
        record=dict(round=i,steps=steps,T=temperature,alpha=.5,
                    teacher_checkpoint_sha256=teacher_hash,scratch=scratch,kd=kd)
        d.write_json(folder/'results_repair.json',record)
        repairs.append(record)
    quant=d.quantize(argparse.Namespace(output=out/'repair_round3',batch_size=24),student)
    for label,a,b in [('baseline',baseline['scratch'],baseline['kd'])]+[(f'repair{i+1}',r['scratch'],r['kd']) for i,r in enumerate(repairs)]:
        checks[label+'_same_initialization']=a['audit']['initial_sha256']==b['audit']['initial_sha256']
        checks[label+'_same_batches']=a['audit']['batch_sha256']==b['audit']['batch_sha256']
        checks[label+'_same_steps']=a['steps']==b['steps']
        checks[label+'_teacher_no_grad']=not b['audit']['teacher_has_grad']
        checks[label+'_ppl_matches']=abs(b['ppl']-math.exp(b['val_loss']))<1e-10
    checks['teacher_unchanged']=teacher_hash==d.state_hash(teacher)
    checks['all_16_ablation_runs']=len(baseline['ablation'])==16
    checks['ablation_same_initialization']=len({r['audit']['initial_sha256'] for r in baseline['ablation']})==1
    checks['ablation_same_batches']=len({r['audit']['batch_sha256'] for r in baseline['ablation']})==1
    checks['int8_reload_exact']=all(q['reload_max_abs_diff']==0 for q in quant)
    final_path=out/'repair_round3'/'kd180.pt'
    loaded=d.make_model(64,7,args.device)
    loaded.load_state_dict(torch.load(final_path,map_location=args.device,weights_only=True)['state_dict'])
    checks['final_student_reload']=abs(d.evaluate(loaded,d.VAL,24,args.device)-repairs[-1]['kd']['val_loss'])<1e-6
    if not all(checks.values()):
        raise AssertionError({k:v for k,v in checks.items() if not v})
    summary=dict(baseline=baseline,repairs=repairs,final_quantization=quant,checks=checks)
    d.write_json(out/'reproduction_results.json',summary)
    rows=[]
    for row in [baseline[k] for k in ['teacher_c160','scratch','kd','fitnet']]+[r[k] for r in repairs for k in ['scratch','kd']]:
        rows.append({k:row[k] for k in ['name','width','params','steps','T','alpha','feature_weight','train_loss','val_loss','ppl','agreement','train_seconds']})
    d.write_csv(out/'results_summary.csv',rows)
    print(f'Completed: {len(checks)} checks passed. Results: {out / "reproduction_results.json"}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    run(parser.parse_args())
