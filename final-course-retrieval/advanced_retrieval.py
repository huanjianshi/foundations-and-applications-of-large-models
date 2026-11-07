"""Chinese retrieval and cross-encoder reranking. No benchmark labels imported."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from collections import Counter
import re,math,time
import numpy as np
from retrieval import ROOT,read_json,rank,fuse,Lexical

QUERY_PREFIX='为这个句子生成表示以用于检索相关文章：'

def tokenize(text):
    text=text.lower()
    result=[]
    for chunk in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+',text):
        if re.fullmatch(r'[a-z0-9]+',chunk):result.append(chunk)
        else:
            result.extend(chunk)
            result.extend(chunk[i:i+2] for i in range(len(chunk)-1))
    return Counter(result)

class BM25:
    def __init__(self,cards,title_weight=2,k1=1.2,b=.75):
        self.counts=[tokenize(c['text']) for c in cards]
        for counts,c in zip(self.counts,cards):
            for term,tf in tokenize(c['title']).items():counts[term]+=title_weight*tf
        self.lengths=np.array([sum(c.values()) for c in self.counts],dtype=float)
        self.average=self.lengths.mean();self.k1=k1;self.b=b
        terms=set().union(*(set(c) for c in self.counts));n=len(cards)
        self.idf={t:math.log(1+(n-sum(t in c for c in self.counts)+.5)/(sum(t in c for c in self.counts)+.5)) for t in terms}
    def scores(self,q):
        out=np.zeros(len(self.counts))
        for term,qtf in tokenize(q).items():
            if term not in self.idf:continue
            tf=np.array([c.get(term,0) for c in self.counts])
            out+=qtf*self.idf[term]*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths/self.average))
        return out

class TextEncoder:
    def __init__(self,path,kind='bge',device='cuda',threads=4):
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(threads);torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA不可用，可用--device cpu明确切换并重新计时。')
        self.torch=torch;self.path=Path(path);self.kind=kind;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        self.model=AutoModel.from_pretrained(self.path,local_files_only=True,use_safetensors=True).to(device).eval()
        self.max_length=128 if kind=='minilm' else 512
        self.dimension=self.model.config.hidden_size
        self.parameters=sum(p.numel() for p in self.model.parameters())
    def encode(self,texts,query=False,instruction=True,batch_size=8):
        if query and instruction and self.kind=='bge':texts=[QUERY_PREFIX+s for s in texts]
        out=[]
        for i in range(0,len(texts),batch_size):
            x=self.tokenizer(texts[i:i+batch_size],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                h=self.model(**x).last_hidden_state
                if self.kind=='bge':v=h[:,0]
                else:
                    mask=x['attention_mask'].unsqueeze(-1).to(h.dtype)
                    v=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
                v=self.torch.nn.functional.normalize(v,p=2,dim=1)
            out.append(v.cpu().numpy())
        return np.concatenate(out) if out else np.empty((0,self.dimension),dtype=np.float32)
    def sync(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

class Reranker:
    def __init__(self,path,device='cuda'):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,use_safetensors=True).to(device).eval()
        self.parameters=sum(p.numel() for p in self.model.parameters())
    def score(self,query,texts,batch_size=8):
        out=[]
        for start in range(0,len(texts),batch_size):
            subset=texts[start:start+batch_size]
            x=self.tokenizer([query]*len(subset),subset,padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
            with self.torch.inference_mode():out.extend(self.model(**x).logits.view(-1).float().cpu().tolist())
        return np.array(out)

def weighted_rrf(lex,dense,ids,lex_weight=.25,depth=10,constant=60):
    # Zero lexical evidence contributes nothing. Only top-depth positive lexical hits vote.
    lo=[i for i in rank(lex,ids) if lex[i]>0][:depth]
    so=rank(dense,ids)[:depth]
    s=np.zeros(len(ids))
    for weight,order in [(lex_weight,lo),(1-lex_weight,so)]:
        for pos,i in enumerate(order,1):s[i]+=weight/(constant+pos)
    # Tail ties keep semantic order, then stable document ID, without label access.
    dr={i:r for r,i in enumerate(rank(dense,ids))}
    return sorted(range(len(ids)),key=lambda i:(-s[i],dr[i],ids[i]))

def rerank_order(base_order,pair_scores,ids,depth):
    selected=base_order[:depth]
    return sorted(selected,key=lambda i:(-float(pair_scores[i]),ids[i]))+base_order[depth:]

def minmax(values):
    values=np.asarray(values,dtype=float)
    width=values.max()-values.min()
    return (values-values.min())/width if width>0 else np.zeros_like(values)

def late_fusion_order(base_order,lex_scores,dense_scores,pair_scores,ids,depth,weights):
    candidates=np.array(base_order[:depth])
    scores=weights[0]*minmax(lex_scores[candidates])+weights[1]*minmax(dense_scores[candidates])+weights[2]*minmax(pair_scores[candidates])
    local=sorted(range(len(candidates)),key=lambda j:(-float(scores[j]),ids[candidates[j]]))
    return [int(candidates[j]) for j in local]+base_order[depth:]

class AdvancedEngine:
    def __init__(self,bge,reranker,cards,config):
        self.encoder=bge;self.reranker=reranker;self.cards=cards;self.ids=[c['id'] for c in cards]
        self.texts=[c['title']+'。'+c['text'] for c in cards];self.config=config
        self.lexical=BM25(cards,title_weight=config['title_weight'])
        self.baseline_lexical=BM25(cards,title_weight=config.get('baseline_bm25_title_weight',config['title_weight']))
        self.matrix=bge.encode(self.texts)
    def search(self,query,method='rerank',k=5):
        if not isinstance(query,str) or not query.strip():raise ValueError('请输入问题。')
        if len(query)>1000:raise ValueError('问题请控制在1000字符内。')
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=len(self.cards):raise ValueError('Top-K超出范围。')
        if method not in ['bm25','bge','fusion','rerank','optimized','prior_only']:raise ValueError('未知检索方法。')
        self.encoder.sync();started=time.perf_counter();lex=(self.baseline_lexical if method=='bm25' else self.lexical).scores(query)
        if method=='bm25':scores=lex;order=rank(scores,self.ids)
        else:
            scores=self.matrix@self.encoder.encode([query],query=True,instruction=self.config['instruction'])[0]
            if method=='bge':order=rank(scores,self.ids)
            else:
                order=weighted_rrf(lex,scores,self.ids,self.config['lex_weight'],self.config['fusion_depth'])
                if method in ['rerank','optimized','prior_only']:
                    candidates=order[:self.config['rerank_depth']]
                    pair=self.reranker.score(query,[self.texts[i] for i in candidates]) if method!='prior_only' else np.zeros(len(candidates))
                    output_scores=np.full(len(self.ids),float('-inf'))
                    if method=='optimized':
                        w=self.config['final_weights'];local=w[0]*minmax(lex[candidates])+w[1]*minmax(scores[candidates])+w[2]*minmax(pair)
                    elif method=='prior_only':local=minmax(lex[candidates])
                    else:local=pair
                    for i,s in zip(candidates,local):output_scores[i]=s
                    order=sorted(candidates,key=lambda i:(-float(output_scores[i]),self.ids[i]))+order[self.config['rerank_depth']:]
                    scores=output_scores
        self.encoder.sync();elapsed=(time.perf_counter()-started)*1000
        return dict(method=method,elapsed_ms=elapsed,zero_lexical=not np.any(lex),results=[dict(**self.cards[i],rank=r,score=float(scores[i]) if np.isfinite(scores[i]) else None) for r,i in enumerate(order[:k],1)])

class KeywordEngine:
    """Recommended lightweight deployment: NumPy/CPU only, no neural weights loaded."""
    def __init__(self,cards,title_weight=3):
        self.cards=cards;self.ids=[c['id'] for c in cards];self.lexical=BM25(cards,title_weight)
    def search(self,query,method='bm25',k=5):
        if method!='bm25':raise ValueError('轻量模式仅支持BM25；完整比较模式请去掉--lexical-only。')
        if not isinstance(query,str) or not query.strip():raise ValueError('请输入问题。')
        if len(query)>1000:raise ValueError('问题请控制在1000字符内。')
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=len(self.cards):raise ValueError('Top-K超出范围。')
        start=time.perf_counter();scores=self.lexical.scores(query);order=rank(scores,self.ids)
        return dict(method=method,elapsed_ms=(time.perf_counter()-start)*1000,zero_lexical=not np.any(scores),results=[dict(**self.cards[i],score=float(scores[i]),rank=r) for r,i in enumerate(order[:k],1)])
