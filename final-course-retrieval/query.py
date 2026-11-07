import argparse,json
from retrieval import ROOT,Encoder,Engine,model_path,read_json
from advanced_retrieval import KeywordEngine,AdvancedEngine,TextEncoder,Reranker
p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--method',choices=['lexical','semantic','hybrid','bm25','bge','fusion','rerank','optimized'],default='bm25');p.add_argument('--k',type=int,default=5);p.add_argument('--model-path');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
if __name__=='__main__':
 a=p.parse_args();cards=read_json(ROOT/'data/cards.json')
 if a.method in ['lexical','semantic','hybrid']:
  cfg=read_json(ROOT/'config.json');e=Engine(Encoder(model_path(a.model_path),cfg['threads']),cards,cfg['rrf_constant'])
 elif a.method=='bm25':e=KeywordEngine(cards,3)
 else:
  cfg=read_json(ROOT/'results_v2/selected_config.json');e=AdvancedEngine(TextEncoder(ROOT/'models/bge-small-zh-v1.5',device=a.device),Reranker(ROOT/'models/bge-reranker-base',device=a.device),cards,cfg)
 print(json.dumps(e.search(a.query,a.method,a.k),ensure_ascii=False,indent=2))
