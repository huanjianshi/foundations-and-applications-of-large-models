import argparse,json
from retrieval import ROOT,Encoder,Engine,model_path,read_json
p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--method',choices=['lexical','semantic','hybrid'],default='hybrid');p.add_argument('--k',type=int,default=5);p.add_argument('--model-path')
if __name__=='__main__':
 a=p.parse_args();cfg=read_json(ROOT/'config.json')
 e=Engine(Encoder(model_path(a.model_path),cfg['threads']),read_json(ROOT/'data/cards.json'),cfg['rrf_constant'])
 print(json.dumps(e.search(a.query,a.method,a.k),ensure_ascii=False,indent=2))
