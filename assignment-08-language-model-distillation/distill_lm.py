# -*- coding: utf-8 -*-
"""Assignment 8: character GPT knowledge distillation, no external data.

Run --mode quick first. --mode complete runs the required experiments and A/B/D/E.
Training uses FP32; quantization and latency comparisons always use CPU.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

SUBJ = ['小明','小红','阿远','书生','渔夫','牧童','旅人','诗人','少年','故人','归客','山僧','孤客','游子']
PLACE = ['江边','山寺','长亭','渡口','竹林','荒村','城楼','野渡','溪桥','松间','篱下','窗前','陌上','舟中']
VERB = ['望见','听见','想起','送去','寻到','梦见','独对','遥看','静听','又逢','恰遇','空忆']
OBJ = ['明月','落花','孤舟','归雁','残灯','细雨','青山','晚钟','白雪','寒江','疏星','暮云','秋草','远帆']
MOOD = ['心中惆怅','不觉天明','欣然归去','泪湿衣襟','长叹一声','默然无语','忽忆旧游','欲语还休','此意难平','一笑而别']
POEMS = [
    '床前明月光，疑是地上霜。举头望明月，低头思故乡。',
    '春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。',
    '白日依山尽，黄河入海流。欲穷千里目，更上一层楼。',
    '千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。',
    '空山不见人，但闻人语响。返景入深林，复照青苔上。',
    '红豆生南国，春来发几枝。愿君多采撷，此物最相思。',
    '独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。',
    '月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。',
    '两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。',
    '朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。',
    '日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。',
    '渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。',
    '寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。',
    '独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
    '清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。',
    '朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。',
]


def build_corpus(n_lines=3000, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append('%s%s%s%s，%s。' % (s,p,v,o,m))
        elif i % 3 == 1:
            out.append('%s在%s%s了一夜%s，%s。' % (s,p,o,rng.choice(['看','听','守','忆']),m))
        else:
            out.append('%s%s道：“%s%s。”%s。' % (s,v,o,rng.choice(['正好','可怜','可惜','依旧']),m))
    out.append('\n')
    out.extend(POEMS)
    return ''.join(out)


class Tokenizer:
    def __init__(self, text):
        self.itos = sorted(set(text))
        self.stoi = {c:i for i,c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


def config(width):
    return dict(n_layer=2,n_head=4,n_embd=width,block_size=96,dropout=0.0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg['n_embd']
        self.n_head = cfg['n_head']
        self.c_attn = nn.Linear(c,3*c)
        self.c_proj = nn.Linear(c,c)
        self.drop = nn.Dropout(cfg['dropout'])
        self.register_buffer('mask',torch.tril(torch.ones(cfg['block_size'],cfg['block_size'])).view(1,1,cfg['block_size'],cfg['block_size']))

    def forward(self, x):
        b,t,c = x.shape
        q,k,v = self.c_attn(x).split(c,dim=2)
        q,k,v = (z.view(b,t,self.n_head,c//self.n_head).transpose(1,2) for z in (q,k,v))
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:t,:t] == 0,float('-inf'))
        att = self.drop(F.softmax(att,dim=-1))
        y = (att @ v).transpose(1,2).contiguous().view(b,t,c)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        c = cfg['n_embd']
        self.fc = nn.Linear(c,4*c)
        self.proj = nn.Linear(4*c,c)
        self.drop = nn.Dropout(cfg['dropout'])

    def forward(self,x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg['n_embd'])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg['n_embd'])
        self.mlp = MLP(cfg)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self,vocab,cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab,cfg['n_embd'])
        self.pos_emb = nn.Embedding(cfg['block_size'],cfg['n_embd'])
        self.drop = nn.Dropout(cfg['dropout'])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])])
        self.ln_f = nn.LayerNorm(cfg['n_embd'])
        self.head = nn.Linear(cfg['n_embd'],vocab,bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self,idx,need_hidden=False):
        _,t = idx.shape
        x = self.drop(self.tok_emb(idx)+self.pos_emb(torch.arange(t,device=idx.device)))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return (logits,x) if need_hidden else logits

    @torch.no_grad()
    def generate(self,idx,generator,new_tokens=60):
        was_training = self.training
        self.eval()
        for _ in range(new_tokens):
            logits = self(idx[:,-self.cfg['block_size']:])[:,-1,:] / 0.9
            values,_ = torch.topk(logits,20)
            logits[logits < values[:,[-1]]] = -float('inf')
            nxt = torch.multinomial(F.softmax(logits,dim=-1),1,generator=generator)
            idx = torch.cat((idx,nxt),dim=1)
        self.train(was_training)
        return idx


def kd_loss(student,teacher,temperature):
    v = student.size(-1)
    return F.kl_div(F.log_softmax(student.reshape(-1,v)/temperature,dim=-1),
                    F.softmax(teacher.detach().reshape(-1,v)/temperature,dim=-1),
                    reduction='batchmean') * temperature**2


def get_batch(data,bs,block,gen,device):
    ix = torch.randint(len(data)-block-1,(bs,),generator=gen)
    x = torch.stack([data[i:i+block] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+1+block] for i in ix]).to(device)
    return x,y


def state_hash(model):
    h = hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model,data,bs,device,iters=20,seed=1234):
    old = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(iters):
        x,y = get_batch(data,bs,96,gen,device)
        losses.append(F.cross_entropy(model(x).reshape(-1,len(TOKENIZER.itos)),y.reshape(-1)).item())
    model.train(old)
    return sum(losses)/len(losses)


@torch.no_grad()
def agreement(student,teacher,data,bs,device,iters=10):
    old_s,old_t = student.training,teacher.training
    student.eval(); teacher.eval()
    gen = torch.Generator().manual_seed(4321)
    hits,total = 0,0
    for _ in range(iters):
        x,_ = get_batch(data,bs,96,gen,device)
        a,b = student(x).argmax(-1),teacher(x).argmax(-1)
        hits += (a == b).sum().item()
        total += b.numel()
    student.train(old_s); teacher.train(old_t)
    return hits/total


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def make_model(width,seed,device):
    torch.manual_seed(seed)
    return GPT(TOKENIZER.vocab_size,config(width)).to(device)


def train(model,data,args,name,steps,teacher=None,T=4.0,alpha=0.5,feat=0.0,cache=None):
    device = args.device
    model.train()
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)
        for p in teacher.parameters():
            p.grad = None
    proj = None
    if feat:
        # Independent projection seed; the student initialization is unchanged.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
            torch.manual_seed(args.seed+19)
            proj = nn.Linear(64,teacher.cfg['n_embd']).to(device)
    params = list(model.parameters())+(list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params,lr=args.lr,weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed+1)
    batch_hash = hashlib.sha256()
    history = []
    initial_hash = state_hash(model)
    sync(device); t0 = time.perf_counter()
    for it in range(steps):
        if cache is None:
            x,y = get_batch(data,args.batch_size,96,gen,device)
            # Hash CPU token IDs, not GPU storage or floating point values.
            batch_hash.update(x.cpu().numpy().tobytes())
            tl = ht = None
            if teacher is not None:
                with torch.no_grad():
                    if feat:
                        tl,ht = teacher(x,need_hidden=True)
                    else:
                        tl = teacher(x)
        else:
            x,y,tl = cache[it%len(cache)]
            batch_hash.update(x.cpu().numpy().tobytes())
            ht = None
        if feat:
            logits,hs = model(x,need_hidden=True)
        else:
            logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1,TOKENIZER.vocab_size),y.reshape(-1))
        kd = kd_loss(logits,tl,T) if tl is not None and alpha else ce.new_zeros(())
        fm = F.mse_loss(proj(hs),ht) if feat else ce.new_zeros(())
        loss = (1-alpha)*ce+alpha*kd+feat*fm if tl is not None else ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{name}: nonfinite loss at step {it+1}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
        opt.step()
        history.append(dict(step=it+1,ce=ce.item(),kd_t2=kd.item(),feature_mse=fm.item(),total_loss=loss.item(),grad_norm=norm.item()))
        if it == 0 or (it+1)%50 == 0 or it+1 == steps:
            print(f'[{name}] {it+1}/{steps} ce={ce.item():.4f} kd={kd.item():.4f} total={loss.item():.4f}',flush=True)
    sync(device)
    elapsed = time.perf_counter()-t0
    return history,elapsed,dict(initial_sha256=initial_hash,batch_sha256=batch_hash.hexdigest(),teacher_has_grad=any(p.grad is not None for p in teacher.parameters()) if teacher is not None else False)


def measure_model(model,history,seconds,audit,args,name,teacher=None,T=None,alpha=0.0,feat=0.0):
    val = evaluate(model,VAL,args.batch_size,args.device)
    train_eval = evaluate(model,TRAIN,args.batch_size,args.device,seed=1234)
    row = dict(name=name,width=model.cfg['n_embd'],params=sum(p.numel() for p in model.parameters()),
               steps=len(history),train_seconds=seconds,train_loss=train_eval,last_batch_ce=history[-1]['ce'],
               val_loss=val,ppl=math.exp(val),agreement=agreement(model,teacher,VAL,args.batch_size,args.device) if teacher is not None else None,
               T=T,alpha=alpha,feature_weight=feat,audit=audit)
    gen = torch.Generator(device=args.device).manual_seed(20261106)
    idx = torch.tensor([TOKENIZER.encode('月')],device=args.device)
    row['sample'] = TOKENIZER.decode(model.generate(idx,gen)[0].tolist())
    write_csv(args.output/f'history_{name}.csv',history)
    write_json(args.output/f'result_{name}.json',row)
    torch.save(dict(state_dict=model.state_dict(),config=model.cfg,vocab=TOKENIZER.itos),args.output/f'{name}.pt')
    print(f'[{name}] val={val:.6f} ppl={math.exp(val):.4f} train_s={seconds:.2f}',flush=True)
    return row


def experiment(args,name,width,steps,teacher=None,T=4.0,alpha=0.5,feat=0.0,cache=None,reference_teacher=None):
    model = make_model(width,args.seed if teacher is None and width != 64 else args.seed+7,args.device)
    h,s,audit = train(model,TRAIN,args,name,steps,teacher,T,alpha,feat,cache)
    return model,measure_model(model,h,s,audit,args,name,teacher if teacher is not None else reference_teacher,T if teacher is not None else None,alpha if teacher is not None else 0.0,feat)


def teacher_model(args,width=160):
    name = f'teacher_c{width}'
    path = args.output/f'{name}.pt'
    if path.exists() and not args.retrain:
        meta = json.loads((args.output/f'result_{name}.json').read_text(encoding='utf-8'))
        saved = torch.load(path,map_location=args.device,weights_only=True)
        if saved['vocab'] != TOKENIZER.itos or saved['config'] != config(width) or meta['steps'] != args.teacher_iters:
            raise ValueError('Teacher cache does not match configuration. Use a new --output or --retrain.')
        source = json.loads((args.output/'run_config.json').read_text(encoding='utf-8'))
        train_digest = hashlib.sha256(build_corpus().encode()).hexdigest()
        if source['seed'] != args.seed or source['lr'] != args.lr or source['batch_size'] != args.batch_size or source['train_sha256'] != train_digest:
            raise ValueError('Teacher cache training settings differ; choose a new output directory.')
        model = make_model(width,args.seed,args.device)
        model.load_state_dict(saved['state_dict'])
        print(f'[{name}] reused; original training time {meta["train_seconds"]:.2f}s',flush=True)
        return model,meta
    return experiment(args,name,width,args.teacher_iters)


def ablate(args,teacher):
    gen = torch.Generator().manual_seed(2024)
    teacher.eval()
    cache = []
    sync(args.device); start = time.perf_counter()
    with torch.no_grad():
        for _ in range(32):
            x,y = get_batch(TRAIN,args.batch_size,96,gen,args.device)
            cache.append((x,y,teacher(x)))
    sync(args.device)
    cache_seconds = time.perf_counter()-start
    rows = []
    for T in [1,2,4,8]:
        for alpha in [0.3,0.5,0.7,0.9]:
            _,row = experiment(args,f'ablate_T{T}_a{alpha}',64,args.abl_iters,teacher,T,alpha,cache=cache)
            rows.append(row)
    write_json(args.output/'results_ablate.json',dict(cache_batches=32,cache_seed=2024,cache_seconds=cache_seconds,ablation=rows))
    return rows


def quantize(args,student):
    torch.backends.quantized.engine = 'x86'
    fp32 = copy.deepcopy(student).to('cpu').eval()
    quant = torch.ao.quantization.quantize_dynamic(fp32,{nn.Linear},dtype=torch.qint8,inplace=False)
    rows = []
    x,_ = get_batch(VAL,args.batch_size,96,torch.Generator().manual_seed(1234),'cpu')
    for name,model in [('fp32',fp32),('int8',quant)]:
        path = args.output/f'student_{name}_cpu.pt'
        torch.save(model.state_dict(),path)
        with torch.no_grad():
            for _ in range(10):
                model(x)
            timings = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(30):
                    model(x)
                timings.append((time.perf_counter()-t0)/30*1000)
        row = dict(precision=name,file_bytes=path.stat().st_size,val_loss=evaluate(model,VAL,args.batch_size,'cpu'),
                   median_forward_ms=sorted(timings)[2],forward_ms_batches=timings,batch_size=args.batch_size,block=96,threads=torch.get_num_threads())
        reloaded = copy.deepcopy(model)
        reloaded.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        with torch.no_grad():
            row['reload_max_abs_diff'] = (model(x)-reloaded(x)).abs().max().item()
        rows.append(row)
    write_json(args.output/'results_quantization.json',rows)
    return rows


def plots(args,res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.unicode_minus':False})
    fig,axs = plt.subplots(1,2,figsize=(10,3.4))
    for name,label in [('teacher_c160','Teacher'),('scratch','Student scratch'),('kd','Student KD')]:
        if name in res:
            with (args.output/f'history_{name}.csv').open(encoding='utf-8-sig') as f:
                h = list(csv.DictReader(f))
            axs[0].plot([int(x['step']) for x in h],[float(x['ce']) for x in h],label=label,lw=1.1)
            if name == 'kd':
                axs[1].plot([int(x['step']) for x in h],[float(x['total_loss']) for x in h],label='KD total',lw=1)
                axs[1].plot([int(x['step']) for x in h],[float(x['kd_t2']) for x in h],label='KL x temperature squared',lw=1)
    for ax,title in zip(axs,['Hard-label cross entropy','KD objective components']):
        ax.set(xlabel='Training step',ylabel='Loss',title=title)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.output/'training_curves.png',dpi=200); plt.close(fig)
    if 'ablation' in res:
        values = [[next(r['val_loss'] for r in res['ablation'] if r['T']==t and r['alpha']==a) for a in [.3,.5,.7,.9]] for t in [1,2,4,8]]
        fig,axs = plt.subplots(1,2,figsize=(9.5,3.5))
        for t,ys in zip([1,2,4,8],values):
            axs[0].plot([.3,.5,.7,.9],ys,'o-',label=f'T={t}',ms=4)
        axs[0].set(xlabel='Alpha',ylabel='Validation cross entropy',title='Temperature / weight ablation')
        axs[0].grid(alpha=.2); axs[0].legend(fontsize=8)
        im=axs[1].imshow(values,cmap='YlGnBu',aspect='auto')
        axs[1].set(xticks=range(4),xticklabels=['0.3','0.5','0.7','0.9'],yticks=range(4),yticklabels=['1','2','4','8'],xlabel='Alpha',ylabel='Temperature',title='Validation cross entropy (80 steps)')
        mid = (max(map(max,values))+min(map(min,values)))/2
        for i,row in enumerate(values):
            for j,val in enumerate(row):
                axs[1].text(j,i,f'{val:.4f}',ha='center',va='center',color='white' if val>mid else 'black',fontsize=9)
        fig.colorbar(im,ax=axs[1],fraction=.05,pad=.03)
        fig.tight_layout(); fig.savefig(args.output/'ablation_heatmap.png',dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['quick','teacher','all','ablate','extensions','complete'],default='all')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs_distill')
    ap.add_argument('--teacher-iters',type=int,default=200)
    ap.add_argument('--student-iters',type=int,default=120)
    ap.add_argument('--abl-iters',type=int,default=80)
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--lr',type=float,default=3e-3)
    ap.add_argument('--T',type=float,default=4.)
    ap.add_argument('--alpha',type=float,default=.5)
    ap.add_argument('--feat',type=float,default=0.)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--retrain',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if args.mode == 'quick':
        args.output = args.output/'quick'
        args.teacher_iters,args.student_iters,args.batch_size = 30,20,8
    args.output.mkdir(parents=True,exist_ok=True)
    global TOKENIZER,TRAIN,VAL
    train_text,val_text = build_corpus(),build_corpus(300,999)
    TOKENIZER = Tokenizer(train_text)
    TRAIN = torch.tensor(TOKENIZER.encode(train_text),dtype=torch.long)
    VAL = torch.tensor(TOKENIZER.encode(val_text),dtype=torch.long)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(vocab=TOKENIZER.vocab_size,train_chars=len(TRAIN),val_chars=len(VAL),
                    train_sha256=hashlib.sha256(train_text.encode()).hexdigest(),val_sha256=hashlib.sha256(val_text.encode()).hexdigest())
    # Validate a cache before replacing its configuration metadata.
    teacher,tr = teacher_model(args)
    write_json(args.output/'run_config.json',settings)
    write_json(args.output/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
               device=args.device,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
               platform=platform.platform(),threads=args.threads,tf32=False))
    res = {'teacher_c160':tr}
    if args.mode == 'teacher':
        write_json(args.output/'results_teacher.json',res)
        return
    if args.mode in ['quick','all','complete','extensions']:
        scratch,sr = experiment(args,'scratch',64,args.student_iters,reference_teacher=teacher)
        kd,kr = experiment(args,'kd',64,args.student_iters,teacher,args.T,args.alpha,args.feat)
        res.update(scratch=sr,kd=kr)
        write_json(args.output/'results_all.json',res)
    if args.mode in ['ablate','complete']:
        res['ablation'] = ablate(args,teacher)
    if args.mode in ['complete','extensions']:
        _,fr = experiment(args,'fitnet',64,args.student_iters,teacher,args.T,args.alpha,0.1)
        res['fitnet']=fr
        res['quantization']=quantize(args,kd)
        scan = []
        for width in [96,160,256]:
            tm,tmr = (teacher,tr) if width==160 else teacher_model(args,width)
            if width==160:
                row=kr
            else:
                _,row=experiment(args,f'kd_teacher_c{width}',64,args.student_iters,tm,args.T,args.alpha)
            scan.append(dict(width=width,teacher=tmr,student=row))
        res['teacher_scan']=scan
        write_json(args.output/'results_extensions.json',dict(fitnet=fr,quantization=res['quantization'],teacher_scan=scan))
    plots(args,res)
    write_json(args.output/'results_complete.json',res)
    allrows = [r for r in res.values() if isinstance(r,dict) and 'params' in r]
    write_csv(args.output/'results.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list)) and k!='sample'} for r in allrows])


if __name__ == '__main__':
    main()
