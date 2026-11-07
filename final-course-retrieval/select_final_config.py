"""Second development stage: compare score fusion AND strong single-branch baselines.
No new-test labels or scores are read. May run only before held-out evaluation.
"""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from retrieval import ROOT,read_json,rank,metrics
from advanced_retrieval import weighted_rrf,late_fusion_order

def norm(a):return (a-a.min())/(a.max()-a.min()) if a.max()>a.min() else np.zeros_like(a)
def main():
 out=ROOT/'results_v2'
 if (out/'new_test_per_question.json').exists():raise SystemExit('Test already exposed; refusing further development selection.')
 if (out/'config_lock_stage1.json').exists():raise SystemExit('Final development selection already completed.')
 qs=read_json(ROOT/'data/questions.json');ids=[c['id'] for c in read_json(ROOT/'data/cards.json')]
 z=np.load(out/'development_score_cache.npz');grid=[]
 for tw in [1,2,3]:
  for instr in [True,False]:
   ds=z['dense_instruction' if instr else 'dense_plain'];ls=z[f'bm25_title{tw}'];ps=z['reranker']
   for depth in [10,20]:
    bases=[weighted_rrf(l,s,ids,.25,10) for l,s in zip(ls,ds)]
    for a in [0,.25,.5,.75,1]:
     for b in [0,.25,.5,.75,1]:
      if a+b>1:continue
      weights=[a,b,1-a-b];vals=[]
      for q,o,l,s,r in zip(qs,bases,ls,ds,ps):
       order=late_fusion_order(o,l,s,r,ids,depth,weights)
       vals.append(metrics([ids[i] for i in order],q['relevant']))
      grid.append(dict(title_weight=tw,instruction=instr,depth=depth,lex=a,dense=b,cross=1-a-b,**{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}))
 grid.sort(key=lambda x:(-x['mrr@5'],-x['recall@5'],x['depth'],not x['instruction'],x['title_weight']))
 best=grid[0]
 baseline=read_json(out/'initial_dev_probe.json')
 assert best['mrr@5']>max(r['mrr@5'] for r in baseline),'Prefer cheaper baseline if score fusion does not improve.'
 with (out/'development_score_fusion.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(best));w.writeheader();w.writerows(grid)
 previous=read_json(out/'config_lock.json');(out/'config_lock_stage1.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')
 cfg=dict(title_weight=best['title_weight'],instruction=best['instruction'],lex_weight=.25,fusion_depth=10,rerank_depth=best['depth'],final_weights=[best['lex'],best['dense'],best['cross']],baseline_bm25_title_weight=3,device='cuda',threads=4,rrf_constant=60,precision='float32',version='2.1')
 (out/'selected_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 previous.update(config_sha256=h(out/'selected_config.json'),selection='Old 60 queries only: MRR@5 then Recall@5; compare pure reranking (120 variants), single-branch baselines, and within-candidate min-max score fusion (180 variants). Stage-1 lock superseded before any new-test evaluation.',configurations_stage1=120,configurations_stage2=len(grid),code_hashes={n:h(ROOT/n) for n in ['advanced_retrieval.py','optimize.py','select_final_config.py']},new_test_evaluated=False,development_best=best,benchmark_policy='Main BM25 baseline uses its own best development title weight 3, not the fusion-selected title weight 1.')
 (out/'config_lock.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('FINAL CONFIG LOCKED',json.dumps(cfg,ensure_ascii=False));print('DEV RESULT',json.dumps(best,ensure_ascii=False))

if __name__=='__main__':main()
