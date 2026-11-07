"""Small CPU retrieval engine. No questions or relevance labels are read here."""
import os
import json
import re
import math
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = ROOT.parent/'tmp'
TMP.mkdir(exist_ok=True)
for key, value in {'TEMP':str(TMP),'TMP':str(TMP),'HF_HOME':str(TMP/'hf'),
                   'MPLCONFIGDIR':str(TMP/'mpl'),'HF_HUB_OFFLINE':'1',
                   'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}.items():
    os.environ[key]=value

METHODS=('lexical','semantic','hybrid')
NAMES={'lexical':'字符TF-IDF','semantic':'语义检索','hybrid':'RRF混合检索'}

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def model_path(value=None):
    value=value or os.getenv('COURSE_MODEL_PATH')
    local=ROOT.parent/'environment/local.json'
    if not value and local.exists(): value=read_json(local).get('model_path')
    if not value or not Path(value).is_dir():
        raise ValueError('模型目录不存在。请设置COURSE_MODEL_PATH或传入--model-path；参见README。')
    return Path(value)

def normalize(text):
    return ''.join(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]+',text.lower()))

def features(text):
    t=normalize(text)
    return Counter(t[i:i+2] for i in range(len(t)-1))

def rank(scores, ids):
    return sorted(range(len(ids)), key=lambda i:(-float(scores[i]),ids[i]))

def fuse(rankings, n, constant=60):
    out=np.zeros(n,dtype=np.float64)
    for order in rankings:
        for r,i in enumerate(order,1): out[i]+=1.0/(constant+r)
    return out

def metrics(order, relevant):
    if not relevant: raise ValueError('Relevant IDs must not be empty for this benchmark.')
    if len(order)!=len(set(order)): raise ValueError('Duplicate retrieved IDs.')
    relevant=set(relevant)
    r={f'recall@{k}':len(set(order[:k])&relevant)/len(relevant) for k in [1,3,5]}
    r['mrr@5']=next((1/i for i,d in enumerate(order[:5],1) if d in relevant),0.0)
    return r

class Lexical:
    def __init__(self,texts):
        counts=[features(s) for s in texts]
        self.vocab=sorted(set().union(*(set(c) for c in counts)))
        self.pos={t:i for i,t in enumerate(self.vocab)}
        self.idf=np.array([math.log((1+len(texts))/(1+sum(t in c for c in counts)))+1 for t in self.vocab])
        self.matrix=np.stack([self.vector(c) for c in counts])
    def vector(self,c):
        v=np.zeros(len(self.vocab))
        for t,n in c.items():
            if t in self.pos: v[self.pos[t]]=n
        v*=self.idf
        norm=np.linalg.norm(v)
        return v/norm if norm else v
    def scores(self,q):
        v=self.vector(features(q))
        return self.matrix@v,not bool(np.any(v))

class Encoder:
    def __init__(self,path,threads=4):
        import torch
        from transformers import AutoTokenizer,AutoModel
        self.torch=torch
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self.path=Path(path)
        self.max_length=read_json(self.path/'sentence_bert_config.json')['max_seq_length']
        pool=read_json(self.path/'1_Pooling/config.json')
        if not pool.get('pooling_mode_mean_tokens') or any(pool.get(k) for k in ['pooling_mode_cls_token','pooling_mode_max_tokens','pooling_mode_mean_sqrt_len_tokens']):
            raise ValueError('This implementation supports the recorded mean-pooling model only.')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        self.model=AutoModel.from_pretrained(self.path,local_files_only=True).cpu().eval()
        self.parameters=sum(p.numel() for p in self.model.parameters())
    def lengths(self,texts):
        return [len(self.tokenizer.encode(s,add_special_tokens=True,truncation=False)) for s in texts]
    def encode(self,texts,batch_size=8):
        if not texts: return np.empty((0,384),dtype=np.float32)
        out=[]
        for start in range(0,len(texts),batch_size):
            x=self.tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
            with self.torch.inference_mode():
                h=self.model(**x).last_hidden_state
                a=x['attention_mask'].unsqueeze(-1).to(h.dtype)
                v=(h*a).sum(1)/a.sum(1).clamp(min=1)
                v=self.torch.nn.functional.normalize(v,p=2,dim=1)
            out.append(v.numpy())
        result=np.concatenate(out)
        if not np.isfinite(result).all(): raise ValueError('Non-finite embedding.')
        return result

class Engine:
    def __init__(self,encoder,cards,constant=60):
        if not cards or len({c['id'] for c in cards})!=len(cards): raise ValueError('Invalid or duplicate cards.')
        self.encoder=encoder;self.cards=cards;self.ids=[c['id'] for c in cards];self.constant=constant
        self.texts=[c['title']+'。'+c['text'] for c in cards]
        self.lexical=Lexical(self.texts)
        self.matrix=encoder.encode(self.texts)
    def search(self,query,method='hybrid',k=5):
        if method not in METHODS: raise ValueError('未知检索方法。')
        if not isinstance(query,str) or not query.strip(): raise ValueError('请输入问题。')
        if len(query)>1000: raise ValueError('问题请控制在1000字符内。')
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=len(self.cards): raise ValueError('Top-K超出范围。')
        zero=False
        if method in ('lexical','hybrid'): lex,zero=self.lexical.scores(query)
        if method in ('semantic','hybrid'): sem=self.matrix@self.encoder.encode([query])[0]
        if method=='lexical': scores=lex
        elif method=='semantic': scores=sem
        else: scores=fuse([rank(lex,self.ids),rank(sem,self.ids)],len(self.ids),self.constant)
        order=rank(scores,self.ids)
        return dict(method=method,zero_lexical=zero,results=[dict(**self.cards[i],score=float(scores[i]),rank=r) for r,i in enumerate(order[:k],1)])
