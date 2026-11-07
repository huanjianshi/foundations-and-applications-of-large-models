"""Reproduce fixed benchmark and plots. Alternate output directories preserve evidence."""
import argparse,csv,hashlib,json,platform,statistics,time,sys
from pathlib import Path
from retrieval import ROOT,Encoder,Engine,METHODS,NAMES,metrics,model_path,read_json,rank,fuse
import numpy as np

def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(out,summary,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc') if Path('C:/Windows/Fonts/msyh.ttc').exists() else None
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#5C7187','#357CA5','#D38A43']
    test=[r for r in summary if r['split']=='test']
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    x=np.arange(4);width=.24
    for i,r in enumerate(test): ax.bar(x+(i-1)*width,[r[k] for k in ['recall@1','recall@3','recall@5','mrr@5']],width,color=colors[i],label=NAMES[r['method']] if font else r['method'])
    ax.set_xticks(x,['Recall@1','Recall@3','Recall@5','MRR@5']);ax.set_ylim(0,1.12);ax.set_ylabel('Score');ax.legend(prop=font,ncol=3,loc='upper center');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(out/'main_results.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    axs[0].bar([r['method'] for r in test],[r['median_query_ms'] for r in test],color=colors);axs[0].set_ylabel('Median query latency (ms)')
    for i,m in enumerate(METHODS):
        rs=[r for r in groups if r['split']=='test' and r['method']==m]
        axs[1].plot([r['type'] for r in rs],[r['mrr@5'] for r in rs],marker='o',label=m,color=colors[i])
    axs[1].set_ylabel('MRR@5');axs[1].set_ylim(0,1.05);axs[1].legend(fontsize=8)
    fig.savefig(out/'latency_and_types.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-path');p.add_argument('--out',type=Path,default=ROOT/'results');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'per_question.json').exists(): raise SystemExit('Use a fresh --out directory to preserve existing results.')
    freeze=read_json(ROOT/'data/freeze.json')
    for name,h in freeze['hashes'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=h: raise SystemExit('Frozen data changed: '+name)
    cfg=read_json(ROOT/'config.json');cards=read_json(ROOT/'data/cards.json');qs=read_json(ROOT/'data/questions.json')
    ids={c['id'] for c in cards}
    if len({q['id'] for q in qs})!=len(qs) or any(not q['relevant'] or not set(q['relevant'])<=ids for q in qs): raise ValueError('Invalid question labels.')
    started=time.perf_counter();enc=Encoder(model_path(args.model_path),cfg['threads']);load=time.perf_counter()-started
    t=time.perf_counter();engine=Engine(enc,cards,cfg['rrf_constant']);build=time.perf_counter()-t
    a=enc.encode(['训练集表现好但测试集差','模型过拟合','词向量表示句子'])
    b=enc.encode(['训练集表现好但测试集差'])
    probe={'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'norms':np.linalg.norm(a,axis=1).tolist(),'batch_max_difference':float(abs(a[0]-b[0]).max())}
    if not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5) or probe['batch_max_difference']>1e-5: raise ValueError('Embedding probe failed.')
    lengths=enc.lengths(engine.texts);qlens=enc.lengths([q['text'] for q in qs])
    # No tests or labels enter the retrieval engine; only plain query strings do.
    for m in METHODS: engine.search('模型如何检索资料',m)
    rows=[];timings=[];full=[]
    for q in qs:
        for m in METHODS:
            times=[];expected=None
            for repeat in range(cfg['repeats']):
                t=time.perf_counter();r=engine.search(q['text'],m,len(cards));ms=(time.perf_counter()-t)*1000
                order=[c['id'] for c in r['results']]
                if expected is not None and expected!=order: raise ValueError('Non-deterministic ranking.')
                expected=order;times.append(ms)
                timings.append(dict(query_id=q['id'],method=m,repeat=repeat+1,query_ms=ms))
            row=dict(query_id=q['id'],split=q['split'],type=q['type'],method=m,**metrics(order,q['relevant']),query_ms=statistics.median(times))
            rows.append(row)
            full.append(dict(**row,query=q['text'],relevant=q['relevant'],ranked_ids=order,scores=[c['score'] for c in r['results']],zero_lexical=r['zero_lexical']))
    keys=['recall@1','recall@3','recall@5','mrr@5']
    summary=[];groups=[]
    for split in ['dev','test']:
        for m in METHODS:
            rr=[r for r in rows if r['split']==split and r['method']==m]
            summary.append(dict(split=split,method=m,n=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in keys},median_query_ms=statistics.median(r['query_ms'] for r in rr)))
            for kind in ['direct','paraphrase','confusion']:
                sub=[r for r in rr if r['type']==kind]
                if sub: groups.append(dict(split=split,method=m,type=kind,n=len(sub),**{k:statistics.mean(r[k] for r in sub) for k in keys}))
    # True single-branch RRF ablations must preserve the branch rankings.
    for q in qs:
        branches={r['method']:r for r in full if r['query_id']==q['id']}
        for m in ['lexical','semantic']:
            order=[engine.ids.index(cid) for cid in branches[m]['ranked_ids']]
            assert [engine.ids[i] for i in rank(fuse([order],len(cards),cfg['rrf_constant']),engine.ids)]==branches[m]['ranked_ids']
    ablation=[dict(variant=label,source_method=m,**{k:r[k] for k in keys}) for label,m in [('full','hybrid'),('without_lexical','semantic'),('without_semantic','lexical')] for r in summary if r['split']=='test' and r['method']==m]
    import importlib.metadata as md
    meta=dict(python=platform.python_version(),system=platform.system(),processor=platform.processor(),device='cpu',threads=cfg['threads'],packages={x:md.version(x) for x in ['torch','transformers','numpy','matplotlib','huggingface-hub']},model_id=cfg['model_id'],revision=cfg['revision'],parameters=enc.parameters,max_length=enc.max_length,model_load_seconds=load,index_seconds=build,card_token_lengths=lengths,query_token_lengths=qlens,truncated_cards=sum(x>enc.max_length for x in lengths),truncated_queries=sum(x>enc.max_length for x in qlens),probe=probe,data_hashes=freeze['hashes'],config=cfg,code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'retrieval.py',Path(__file__)]},annotation='CODEX-authored and not yet reviewed by student',comparison_status='fixed configuration; no post-test tuning')
    dump(args.out/'per_question.json',full);dump(args.out/'summary.json',summary);dump(args.out/'metadata.json',meta);dump(args.out/'ablation.json',ablation)
    csv_write(args.out/'per_question.csv',rows);csv_write(args.out/'summary.csv',summary);csv_write(args.out/'query_timings.csv',timings);csv_write(args.out/'by_type.csv',groups);csv_write(args.out/'ablation.csv',ablation)
    plot(args.out,summary,groups)
    print(json.dumps({'summary':summary,'probe':probe,'truncated_cards':meta['truncated_cards']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
