"""Development-only model/configuration selection, followed by locked new-test evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,csv,time,statistics,platform
from pathlib import Path
import numpy as np
from retrieval import ROOT,read_json,model_path,Lexical,rank,fuse,metrics,Engine
from advanced_retrieval import BM25,TextEncoder,Reranker,weighted_rrf,rerank_order,AdvancedEngine,late_fusion_order

OUT=ROOT/'results_v2'
METHOD_NAMES={'tfidf':'原字符TF-IDF','minilm':'原MiniLM','rrf_v1':'原完整排名RRF','bm25':'改进BM25','bge':'中文BGE','fusion':'候选排名融合','rerank':'纯神经重排','optimized':'词法先验融合重排','prior_only':'移除神经重排分数','rerank_bge_only':'仅BGE候选＋重排'}
METHODS=list(METHOD_NAMES)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean_metrics(orders,qs,ids):
 return {k:statistics.mean(metrics([ids[i] for i in order],q['relevant'])[k] for order,q in zip(orders,qs)) for k in ['recall@1','recall@3','recall@5','mrr@5']}
def load_models(device):
 print('Loading BGE and reranker',flush=True)
 bge=TextEncoder(ROOT/'models/bge-small-zh-v1.5',device=device)
 rr=Reranker(ROOT/'models/bge-reranker-base',device=device)
 return bge,rr
def cache_scores(qs,cards,bge,rr):
 texts=[c['title']+'。'+c['text'] for c in cards]
 docs=bge.encode(texts);inputs=[q['text'] for q in qs]
 scores={True:bge.encode(inputs,query=True,instruction=True)@docs.T,False:bge.encode(inputs,query=True,instruction=False)@docs.T}
 pairs=[]
 for i,q in enumerate(qs):
  pairs.append(rr.score(q['text'],texts))
  if (i+1)%10==0:print(f'Pair scoring {i+1}/{len(qs)}',flush=True)
 return scores,np.stack(pairs)

def develop(device):
 OUT.mkdir(exist_ok=True)
 if (OUT/'config_lock.json').exists():raise SystemExit('Configuration already locked; do not retune after test exposure.')
 qs=read_json(ROOT/'data/questions.json');cards=read_json(ROOT/'data/cards.json');ids=[c['id'] for c in cards]
 assert len(qs)==60
 # New test is never opened, only its bytes/hash are recorded here.
 f=read_json(ROOT/'data/v2/test_freeze.json');assert sha(ROOT/'data/v2/new_test.json')==f['query_sha256']
 bge,rr=load_models(device);dense,pair=cache_scores(qs,cards,bge,rr)
 lexical={w:np.stack([BM25(cards,w).scores(q['text']) for q in qs]) for w in [1,2,3]}
 grid=[]
 for tw in [1,2,3]:
  for instr in [True,False]:
   for weight in [0,.25,.5,.75,1]:
    for depth in [10,20]:
     base=[weighted_rrf(l,s,ids,weight,depth) for l,s in zip(lexical[tw],dense[instr])]
     for rd in [10,20]:
      orders=[rerank_order(o,p,ids,rd) for o,p in zip(base,pair)]
      grid.append(dict(title_weight=tw,instruction=instr,lex_weight=weight,fusion_depth=depth,rerank_depth=rd,**mean_metrics(orders,qs,ids)))
 # Fixed pre-test selection objective: MRR@5, then Recall@5; tie goes to fewer reranked candidates,
 # then the official query instruction and balanced lexical support, not test performance.
 ordered=sorted(grid,key=lambda r:(-r['mrr@5'],-r['recall@5'],r['rerank_depth'],not r['instruction'],abs(r['lex_weight']-.25),r['title_weight'],r['fusion_depth']))
 best=ordered[0]
 cfg={k:best[k] for k in ['title_weight','instruction','lex_weight','fusion_depth','rerank_depth']}
 cfg.update(device=device,threads=4,rrf_constant=60,precision='float32',version='2.0')
 dump(OUT/'selected_config.json',cfg);write_csv(OUT/'development_grid.csv',ordered)
 base=[]
 for instr in [True,False]:base.append(dict(method='BGE_instruction' if instr else 'BGE_without_instruction',**mean_metrics([rank(x,ids) for x in dense[instr]],qs,ids)))
 for w in [1,2,3]:base.append(dict(method=f'BM25_title{w}',**mean_metrics([rank(x,ids) for x in lexical[w]],qs,ids)))
 write_csv(OUT/'development_baselines.csv',base)
 np.savez_compressed(OUT/'development_score_cache.npz',dense_instruction=dense[True],dense_plain=dense[False],reranker=pair,**{f'bm25_title{w}':v for w,v in lexical.items()})
 lock=dict(config_sha256=sha(OUT/'selected_config.json'),old_queries_sha256=sha(ROOT/'data/questions.json'),new_test_sha256=f['query_sha256'],cards_sha256=sha(ROOT/'data/cards.json'),selection='MRR@5 then Recall@5 on old 60 queries; deterministic efficiency/preference tie-break; no new-test scores accessed.',configurations=len(grid),model_records=read_json(ROOT.parent/'environment/optimization_models.json'),code_hashes={n:sha(ROOT/n) for n in ['advanced_retrieval.py','optimize.py']},new_test_evaluated=False)
 dump(OUT/'config_lock.json',lock)
 print('LOCKED',json.dumps(best,ensure_ascii=False),flush=True)
 print('DEV BASELINES',json.dumps(base,ensure_ascii=False),flush=True)

def evaluate(device,output=None):
 lock=read_json(OUT/'config_lock.json');cfg=read_json(OUT/'selected_config.json')
 target=Path(output) if output else OUT
 target.mkdir(parents=True,exist_ok=True)
 assert sha(OUT/'selected_config.json')==lock['config_sha256']
 assert sha(ROOT/'data/v2/new_test.json')==lock['new_test_sha256'] and sha(ROOT/'data/cards.json')==lock['cards_sha256']
 if (target/'new_test_per_question.json').exists():raise SystemExit('Output already exists. Use a fresh --out directory; no retuning is performed.')
 qs=read_json(ROOT/'data/v2/new_test.json');cards=read_json(ROOT/'data/cards.json');ids=[c['id'] for c in cards];texts=[c['title']+'。'+c['text'] for c in cards]
 bge,rr=load_models(device);dense,pair=cache_scores(qs,cards,bge,rr)
 mini=TextEncoder(model_path(),kind='minilm',device=device)
 old_dense=mini.encode([q['text'] for q in qs])@mini.encode(texts).T
 tfidf=Lexical(texts);bm=BM25(cards,cfg['title_weight']);bm_base=BM25(cards,cfg['baseline_bm25_title_weight']);lex=[tfidf.scores(q['text'])[0] for q in qs];bm_scores=[bm.scores(q['text']) for q in qs]
 all_orders={m:[] for m in METHODS};candidate_recall=[]
 for q,l,old,b,s,p in zip(qs,lex,old_dense,bm_scores,dense[cfg['instruction']],pair):
  ol=rank(l,ids);od=rank(old,ids);bo=rank(bm_base.scores(q['text']),ids);so=rank(s,ids)
  fo=weighted_rrf(b,s,ids,cfg['lex_weight'],cfg['fusion_depth'])
  orders=[ol,od,rank(fuse([ol,od],len(ids),60),ids),bo,so,fo,rerank_order(fo,p,ids,cfg['rerank_depth']),late_fusion_order(fo,b,s,p,ids,cfg['rerank_depth'],cfg['final_weights']),late_fusion_order(fo,b,s,p,ids,cfg['rerank_depth'],[1,0,0]),rerank_order(so,p,ids,cfg['rerank_depth'])]
  for method,order in zip(METHODS,orders):all_orders[method].append(order)
  candidate_recall.append(len(set(ids[i] for i in fo[:cfg['rerank_depth']])&set(q['relevant']))/len(q['relevant']))
 # Latency sample declared without consulting scores: first five cards, all three query types (15 queries).
 advanced=AdvancedEngine(bge,rr,cards,cfg);old_engine=Engine(mini,cards)
 samples=qs[:15];times=[]
 def search(q,method):
  if method in ['tfidf','minilm','rrf_v1']:return old_engine.search(q,{'tfidf':'lexical','minilm':'semantic','rrf_v1':'hybrid'}[method],len(cards))
  if method=='rerank_bge_only':
   s=advanced.matrix@bge.encode([q],query=True,instruction=cfg['instruction'])[0];so=rank(s,ids)
   ps=rr.score(q,[texts[i] for i in so[:cfg['rerank_depth']]])
   ro=sorted(zip(so[:cfg['rerank_depth']],ps),key=lambda v:(-v[1],ids[v[0]]))
   return {'results':[dict(id=ids[i]) for i,_ in ro]+[dict(id=ids[i]) for i in so[cfg['rerank_depth']:]]}
  return advanced.search(q,method,len(cards))
 for method in METHODS:
  search('什么是检索排名融合？',method);bge.sync()
  for qi,q in enumerate(samples):
   expected=[ids[i] for i in all_orders[method][qi]]
   for rep in range(3):
    bge.sync();t=time.perf_counter();r=search(q['text'],method);bge.sync();ms=(time.perf_counter()-t)*1000
    actual=[c['id'] for c in r['results']]
    assert actual==expected,('Batch/single-query ranking mismatch',method,q['id'],actual,expected)
    times.append(dict(query_id=q['id'],method=method,repeat=rep+1,query_ms=ms))
 print('Latency and ranking consistency passed',flush=True)
 rows=[];summaries=[];by_type=[]
 for method in METHODS:
  s=mean_metrics(all_orders[method],qs,ids)
  median=statistics.median(statistics.median(r['query_ms'] for r in times if r['method']==method and r['query_id']==q['id']) for q in samples)
  summaries.append(dict(method=method,name=METHOD_NAMES[method],n=90,**s,median_query_ms=median,latency_queries=15))
  for q,o in zip(qs,all_orders[method]):rows.append(dict(query_id=q['id'],type=q['type'],method=method,query=q['text'],relevant=q['relevant'],ranked_ids=[ids[i] for i in o],**metrics([ids[i] for i in o],q['relevant'])))
  for kind in ['term','description','contrast']:
   sub=[(q,o) for q,o in zip(qs,all_orders[method]) if q['type']==kind]
   by_type.append(dict(method=method,type=kind,n=len(sub),**mean_metrics([o for q,o in sub],[q for q,o in sub],ids)))
 dump(target/'new_test_per_question.json',rows);dump(target/'new_test_summary.json',summaries);write_csv(target/'new_test_summary.csv',summaries);write_csv(target/'latency_samples.csv',times);write_csv(target/'new_test_by_type.csv',by_type)
 np.savez_compressed(target/'new_test_score_cache.npz',dense=dense[cfg['instruction']],reranker=pair,old_minilm=old_dense)
 import torch,importlib.metadata as md
 meta=dict(test_queries=90,development_queries=60,test_label_review='CODEX authored; not yet individually human audited',device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,threads=4,precision='float32',versions={p:md.version(p) for p in ['torch','transformers','numpy']},model_parameters={'minilm':mini.parameters,'bge':bge.parameters,'reranker':rr.parameters},config=cfg,lock=lock,candidate_recall_at_selected_depth=statistics.mean(candidate_recall),latency='15 predefined questions (first five cards × 3 types), 3 repeats, per-query median then median across queries; GPU synchronized; no model load/index in query timer',timing_rows=len(times),all_sampled_rankings_consistent=True,new_test_tuning=False)
 meta['reproduction_device_differs_from_selection']=device!=cfg['device']
 meta['executed_code_hashes']={n:sha(ROOT/n) for n in ['advanced_retrieval.py','optimize.py']}
 dump(target/'new_test_metadata.json',meta)
 print(json.dumps({'summary':summaries,'candidate_recall':meta['candidate_recall_at_selected_depth']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['dev','test']);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--out',type=Path);a=p.parse_args()
 develop(a.device) if a.mode=='dev' else evaluate(a.device,a.out)
