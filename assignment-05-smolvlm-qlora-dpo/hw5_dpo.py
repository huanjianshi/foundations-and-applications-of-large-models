# -*- coding: utf-8 -*-
"""HW5: synthetic visual QA, NF4 QLoRA SFT and hand-written DPO.

Run with NO5/.venv/Scripts/python.exe. All paths are relative to this file.
Reference policy: frozen quantized base with the adapter disabled (guide §4.3).
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTHONUTF8', '1')
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
COLORS = {'红': (220, 50, 50), '蓝': (50, 80, 220),
          '绿': (40, 170, 70), '黄': (230, 190, 40)}
IMG_SPEC = {
    'img01': [('红', '圆', 2), ('蓝', '方', 1)],
    'img02': [('绿', '圆', 3)],
    'img03': [('蓝', '圆', 1), ('黄', '方', 2)],
    'img04': [('红', '方', 3)],
    'img05': [('绿', '方', 2), ('红', '圆', 1)],
    'img06': [('黄', '圆', 4)],
    'img07': [('蓝', '方', 2), ('绿', '圆', 1)],
    'img08': [('红', '圆', 4)],
    'img09': [('绿', '方', 1), ('黄', '圆', 2)],
    'img10': [('蓝', '圆', 3)],
    'img11': [('黄', '方', 1)],
    'img12': [('红', '方', 1), ('蓝', '方', 1)],
    'img13': [('绿', '圆', 2), ('黄', '圆', 1)],
    'img14': [('蓝', '方', 3), ('红', '方', 1)],
    'img15': [('黄', '圆', 2), ('绿', '圆', 2)],
}
QUESTIONS = [
    {'id': 'P01', 'type': 'count', 'img': 'img01', 'q': '图中有几个红色圆形？', 'gold': '2'},
    {'id': 'P02', 'type': 'count', 'img': 'img02', 'q': '图中有几个绿色圆形？', 'gold': '3'},
    {'id': 'P03', 'type': 'count', 'img': 'img03', 'q': '图中有几个黄色方形？', 'gold': '2'},
    {'id': 'P04', 'type': 'count', 'img': 'img06', 'q': '图中有几个黄色圆形？', 'gold': '4'},
    {'id': 'P05', 'type': 'count', 'img': 'img10', 'q': '图中有几个蓝色圆形？', 'gold': '3'},
    {'id': 'P06', 'type': 'attr', 'img': 'img01', 'q': '图中蓝色的图形是什么形状？', 'gold': '方形'},
    {'id': 'P07', 'type': 'attr', 'img': 'img05', 'q': '图中红色的图形是什么形状？', 'gold': '圆形'},
    {'id': 'P08', 'type': 'attr', 'img': 'img09', 'q': '图中绿色的图形是什么形状？', 'gold': '方形'},
    {'id': 'P09', 'type': 'attr', 'img': 'img13', 'q': '图中黄色的图形是什么形状？', 'gold': '圆形'},
    {'id': 'P10', 'type': 'attr', 'img': 'img14', 'q': '图中蓝色的图形是什么形状？', 'gold': '方形'},
    {'id': 'P11', 'type': 'exist', 'img': 'img11', 'q': '图里有没有黄色方形？', 'gold': '有'},
    {'id': 'P12', 'type': 'exist', 'img': 'img12', 'q': '图里有没有红色圆形？', 'gold': '没有'},
    {'id': 'P13', 'type': 'exist', 'img': 'img07', 'q': '图里有没有绿色圆形？', 'gold': '有'},
    {'id': 'P14', 'type': 'exist', 'img': 'img08', 'q': '图里有没有绿色方形？', 'gold': '没有'},
    {'id': 'P15', 'type': 'exist', 'img': 'img15', 'q': '图里有没有蓝色图形？', 'gold': '没有'},
]
SFT_DATA = [dict(img=q['img'], prompt=q['q'],
                 answer=q['gold'] + ('个。' if q['type'] == 'count' else '。'))
            for q in QUESTIONS]
PRE_PAIRS = [
    ('img01', '图中有几个红色圆形？', '2 个。', '3 个。图中有三个红色圆形整齐排列。'),
    ('img02', '图中有几个绿色圆形？', '3 个。', '2 个。'),
    ('img03', '图中有几个黄色方形？', '2 个。', '没有看到黄色图形。'),
    ('img06', '图中有几个黄色圆形？', '4 个。', '3 个。'),
    ('img10', '图中有几个蓝色圆形？', '3 个。', '图中有三个蓝色方形。'),
    ('img01', '图中蓝色的图形是什么形状？', '方形。', '圆形。'),
    ('img05', '图中红色的图形是什么形状？', '圆形。', '方形，而且有两个红色方形。'),
    ('img09', '图中绿色的图形是什么形状？', '方形。', '图中没有绿色图形。'),
    ('img11', '图里有没有黄色方形？', '有。', '没有。'),
    ('img12', '图里有没有红色圆形？', '没有。', '有，图左上角有一个红色圆形。'),
    ('img08', '图里有没有绿色方形？', '没有。', '有，一共两个绿色方形。'),
    ('img15', '图里有没有蓝色图形？', '没有。图中只有黄色和绿色的圆形。', '有，两个蓝色圆形。'),
]
EXTRA_PAIRS = [
    ('img04', '图中有几个红色方形？', '3个。', '2个。', '计数少一'),
    ('img05', '图中绿色的图形是什么形状？', '方形。', '圆形。', '颜色与形状绑定错误'),
    ('img11', '图里有没有蓝色圆形？', '没有。', '有，一个蓝色圆形。', '不存在物体的肯定幻觉'),
    ('img12', '图里有没有蓝色方形？', '有。', '没有。图中没有蓝色物体。', '存在物体的否定偏置'),
    ('img15', '图中有几个黄色圆形？', '2个。', '2个。它们正在向右移动，并准备参加比赛。', '答案正确但理由编造'),
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_images():
    folder = ROOT / 'hw5_imgs'
    folder.mkdir(exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        image = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(image)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                size = rnd.randint(30, 50)
                box = [x, y, x + size, y + size]
                getattr(draw, 'ellipse' if shape == '圆' else 'rectangle')(
                    box, outline=COLORS[color], width=4)
        image.save(folder / f'{img_id}.png')
    dump(ROOT / 'results' / 'data_manifest.json', {
        'image_seed': 42, 'img_spec': IMG_SPEC, 'questions': QUESTIONS,
        'sft': SFT_DATA, 'pre_pairs': PRE_PAIRS, 'extra_pairs': EXTRA_PAIRS,
        'image_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(folder.glob('*.png'))}})
    print('[data] generated 15 images, guide seed=42', flush=True)


def image(img_id):
    path = ROOT / 'hw5_imgs' / f'{img_id}.png'
    if not path.exists():
        make_images()
    with Image.open(path) as im:
        return im.convert('RGB')


def pairs(version):
    rows = [dict(img=p[0], prompt=p[1], chosen=p[2], rejected=p[3], source='guide')
            for p in PRE_PAIRS]
    if version == 'expanded':
        rows += [dict(img=p[0], prompt=p[1], chosen=p[2], rejected=p[3],
                      source='added', failure=p[4]) for p in EXTRA_PAIRS]
    return rows


def load_model(args, train=False, adapter=None):
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('This full experiment requires CUDA; do not silently substitute CPU.')
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    # A 256px synthetic image does not need the default 2048px/17-patch expansion.
    # Same preprocessing in all stages: one 512px vision patch, no image splitting.
    processor.image_processor.size = {'longest_edge': 512}
    processor.image_processor.do_image_splitting = False
    kwargs = dict(local_files_only=True, torch_dtype=torch.float32,
                  attn_implementation='sdpa')
    if train:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float32)
        kwargs['device_map'] = {'': 0}
    base = AutoModelForVision2Seq.from_pretrained(args.model_path, **kwargs)
    if train:
        base = prepare_model_for_kbit_training(base,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        base.config.use_cache = False
        if adapter:
            model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        else:
            config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type='CAUSAL_LM')
            model = get_peft_model(base, config)
    else:
        base.to('cuda')
        model = PeftModel.from_pretrained(base, str(adapter)) if adapter else base
    return processor, model


def build_prompt(processor, question):
    messages = [{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def encode_answer(processor, img, prompt, response):
    text = build_prompt(processor, prompt)
    prefix = processor(text=[text], images=[img], return_tensors='pt')
    full = processor(text=[text + response], images=[img], return_tensors='pt')
    plen = prefix['input_ids'].shape[1]
    if not torch.equal(prefix['input_ids'], full['input_ids'][:, :plen]):
        raise ValueError('Prompt token boundary changed; refusing an incorrect answer mask.')
    labels = full['input_ids'].clone()
    labels[:, :plen] = -100
    labels[full['attention_mask'] == 0] = -100
    assert labels[:, plen:].ne(-100).any()
    assert (labels[:, :plen] == -100).all()
    full['labels'] = labels
    return full, plen


def seq_logprob(model, batch):
    inputs = {k: v for k, v in batch.items() if k != 'labels'}
    logits = model(**inputs).logits[:, :-1].float()
    targets = batch['labels'][:, 1:]
    mask = targets.ne(-100)
    # Sum answer-token log probabilities, not their mean.
    token_lp = logits.log_softmax(-1).gather(
        -1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return token_lp[mask].sum()


def dpo_loss(pol_w, ref_w, pol_l, ref_l, beta):
    margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
    return -F.logsigmoid(margin), margin


def dpo_check(arg):
    pw, rw, pl, rl, beta = map(float, arg.split(','))
    margin = beta * ((pw - rw) - (pl - rl))
    loss = max(-margin, 0) + math.log1p(math.exp(-abs(margin)))
    print(f'margin={margin:.8f} loss={loss:.8f}', flush=True)
    return dict(pol_w=pw, ref_w=rw, pol_l=pl, ref_l=rl, beta=beta, margin=margin, loss=loss)


def metrics(question, prediction):
    p = prediction.replace(' ', '')
    guide = question['gold'] in p  # Exact guide substring rule, including its false positives.
    low = prediction.strip().lower()
    if question['type'] == 'count':
        numbers = set(re.findall(r'\d+', low))
        for word, num in [('one', '1'), ('two', '2'), ('three', '3'), ('four', '4'),
                          ('一个', '1'), ('两个', '2'), ('三个', '3'), ('四个', '4')]:
            if re.search(r'\b' + word + r'\b', low) if word.isascii() else word in low:
                numbers.add(num)
        audited = numbers == {question['gold']}
    elif question['type'] == 'attr':
        square = bool(re.search(r'方形|正方|矩形|square|rectangle', low))
        circle = bool(re.search(r'圆形|圆圈|circle|circular|round', low))
        audited = (square and not circle) if question['gold'] == '方形' else (circle and not square)
    else:
        neg = bool(re.search(r'没有|不存在|未见|看不到|无此|\bno\b|\bnot\b|\bnone\b', low))
        pos = bool(re.search(r'\byes\b|there (?:is|are)|存在|有', low))
        audited = neg if question['gold'] == '没有' else (pos and not neg)
        if neg and re.search(r'\byes\b', low):
            audited = False  # Contradictory Yes/No responses are not a clear answer.
        color_words = {'红': ('红', 'red'), '蓝': ('蓝', 'blue'),
                       '绿': ('绿', 'green'), '黄': ('黄', 'yellow')}
        for color, words in color_words.items():
            if color in question['q']:
                other_words = [w for c, ws in color_words.items() if c != color for w in ws]
                if any(w in low for w in other_words) and not any(w in low for w in words):
                    audited = False  # Reject a response answering about a different color.
    return int(guide), int(audited)


@torch.no_grad()
def answer(processor, model, img, question, max_new=24):
    model.eval()
    batch = processor(text=[build_prompt(processor, question)], images=[img],
                      return_tensors='pt').to('cuda')
    ids = model.generate(**batch, max_new_tokens=max_new, do_sample=False, use_cache=True)
    return processor.decode(ids[0, batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def sync():
    torch.cuda.synchronize()


def environment(args):
    import psutil
    return dict(python=sys.version, packages={p: importlib.metadata.version(p) for p in
        ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'pillow']},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        ram_gib=psutil.virtual_memory().total / 2**30,
        cli={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        image_seed=42, seed=args.seed, input_image_size=256,
        processor_longest_edge=512, image_splitting=False,
        evaluation_precision='fp32', training_precision='NF4/fp32 compute',
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def evaluate(args):
    started = time.perf_counter()
    processor, model = load_model(args, adapter=args.adapter)
    torch.cuda.reset_peak_memory_stats()
    sync()
    eval_start = time.perf_counter()
    rows = []
    questions = QUESTIONS if args.question_set == 'original' else [
        dict(id=f'N{i+1:02d}', type='count' if '几个' in p[1] else
             'attr' if '形状' in p[1] else 'exist', img=p[0], q=p[1],
             gold=['3', '方形', '没有', '有', '2'][i])
        for i, p in enumerate(EXTRA_PAIRS)]
    for q in questions:
        pred = answer(processor, model, image(q['img']), q['q'], args.max_new_tokens)
        guide, audited = metrics(q, pred)
        rows.append(dict(stage=args.tag, **q, prediction=pred,
                         guide_correct=guide, audited_correct=audited))
        print(f"[{args.tag} {q['id']}] {pred!r} | guide={guide} audited={audited}", flush=True)
    sync()
    stats = environment(args)
    stats.update(inference_seconds=time.perf_counter() - eval_start,
                 elapsed_seconds=time.perf_counter() - started,
                 peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30)
    summary = []
    for tp in ['count', 'attr', 'exist', 'total']:
        subset = rows if tp == 'total' else [r for r in rows if r['type'] == tp]
        summary.append(dict(stage=args.tag, type=tp, n=len(subset),
            guide_correct=sum(r['guide_correct'] for r in subset),
            audited_correct=sum(r['audited_correct'] for r in subset)))
    write_csv(args.output / 'predictions.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    dump(args.output / 'metadata.json', stats)
    print('[eval summary]', summary, flush=True)


def train(args):
    started = time.perf_counter()
    processor, model = load_model(args, train=True,
        adapter=args.sft_adapter if args.mode == 'dpo' else None)
    trainable, total = model.get_nb_trainable_parameters()
    parameter_stats = dict(trainable=trainable, total=total, percent=100 * trainable / total)
    model.print_trainable_parameters()
    targets = [n for n, m in model.named_modules() if hasattr(m, 'lora_A')]
    dump(args.output / 'parameter_stats.json', dict(**parameter_stats, target_modules=targets))
    metadata = environment(args)
    data = SFT_DATA if args.mode == 'sft' else pairs(args.pairs)
    encoded = []
    for item in data:
        responses = [item['answer']] if args.mode == 'sft' else [item['chosen'], item['rejected']]
        encoded.append([encode_answer(processor, image(item['img']), item['prompt'], resp)[0]
                        for resp in responses])
    mask_sample, plen = encode_answer(processor, image(data[0]['img']), data[0]['prompt'],
        data[0]['answer'] if args.mode == 'sft' else data[0]['chosen'])
    dump(args.output / 'mask_check.json', dict(prompt_length=plen,
        full_length=mask_sample['input_ids'].shape[1],
        supervised_tokens=int(mask_sample['labels'].ne(-100).sum()),
        supervised_text=processor.tokenizer.decode(mask_sample['labels'][mask_sample['labels'].ne(-100)]),
        prefix_identical=True, prompt_masked=True))
    reference = []
    ref_started = time.perf_counter()
    if args.mode == 'dpo':
        # No stochastic dropout in preference log probabilities (policy or reference).
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            for chosen, rejected in encoded:
                rw = float(seq_logprob(model, chosen.to('cuda')))
                rl = float(seq_logprob(model, rejected.to('cuda')))
                reference.append((rw, rl))
        dump(args.output / 'reference_logprobs.json', reference)
    sync()
    reference_seconds = time.perf_counter() - ref_started
    model.train()
    lr = 1e-4 if args.mode == 'sft' else 5e-5
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = args.steps if args.steps is not None else (60 if args.mode == 'sft' else 40)
    torch.cuda.reset_peak_memory_stats()
    sync()
    training_start = time.perf_counter()
    history = []
    step, epoch = 0, 0
    while step < steps:
        order = list(range(len(data)))
        random.Random(epoch).shuffle(order)
        for idx in order:
            if step == steps:
                break
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if args.mode == 'sft':
                loss = model(**encoded[idx][0].to('cuda')).loss
                row = dict(step=step + 1, epoch=epoch + 1, sample_index=idx, loss=float(loss.detach()))
            else:
                chosen, rejected = encoded[idx]
                pw = seq_logprob(model, chosen.to('cuda'))
                pl = seq_logprob(model, rejected.to('cuda'))
                rw, rl = reference[idx]
                loss, margin = dpo_loss(pw, rw, pl, rl, args.beta)
                row = dict(step=step + 1, epoch=epoch + 1, sample_index=idx,
                    loss=float(loss.detach()), margin=float(margin.detach()),
                    pol_w=float(pw.detach()), ref_w=rw, pol_l=float(pl.detach()), ref_l=rl)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step + 1}')
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Missing or non-finite adapter gradient')
            row['grad_norm'] = math.sqrt(sum(float(g.float().square().sum()) for g in grads))
            optimizer.step()
            sync()
            row['seconds'] = time.perf_counter() - tick
            history.append(row)
            step += 1
            print(f'[{args.mode}] step={step}/{steps} loss={row["loss"]:.6f}' +
                  (f' margin={row["margin"]:.6f}' if args.mode == 'dpo' else '') +
                  f' seconds={row["seconds"]:.2f}', flush=True)
        epoch += 1
    sync()
    metadata.update(**parameter_stats, actual_steps=step, learning_rate=lr,
        batch_size=1, samples=len(data), reference_cache_seconds=reference_seconds,
        training_seconds=time.perf_counter() - training_start,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
        elapsed_seconds=time.perf_counter() - started,
        dpo_dropout=0.0 if args.mode == 'dpo' else None,
        optimizer='AdamW (not paged)', frozen_base_gradients_absent=True)
    model.save_pretrained(args.output / 'adapter')
    processor.save_pretrained(args.output / 'adapter')
    write_csv(args.output / 'history.csv', history)
    dump(args.output / 'metadata.json', metadata)
    assert len(history) == steps
    print('[saved]', args.output, flush=True)


def verify():
    checks = [dpo_check(s) for s in ['1.2,0.8,0.9,0.6,0.1',
              '-1,-1,-2,-2,0.1', '-2,-1,-1,-1,0.1', '1.5,1.0,0.2,0.6,0.1']]
    assert abs(checks[1]['loss'] - math.log(2)) < 1e-12
    assert checks[0]['loss'] < checks[1]['loss'] < checks[2]['loss']
    pw = torch.tensor(-2.0, requires_grad=True)
    pl = torch.tensor(-3.0, requires_grad=True)
    loss, margin = dpo_loss(pw, -2., pl, -3., .1)
    loss.backward()
    assert pw.grad < 0 and pl.grad > 0
    assert metrics(QUESTIONS[10], '没有。') == (1, 0)
    dump(ROOT / 'results' / 'dpo_checks.json', dict(cases=checks,
        chosen_gradient=float(pw.grad), rejected_gradient=float(pl.grad),
        gradient_direction_correct=True, guide_exist_false_positive_reproduced=True))
    print('[verify] DPO arithmetic, gradient direction and scoring checks PASS')


def run_all(args):
    verify()
    make_images()
    result = ROOT / 'results'
    result.mkdir(exist_ok=True)
    def run(*command):
        tag = command[command.index('--output') + 1] if '--output' in command else 'run'
        log = result / (Path(tag).name + '.log')
        cmd = [sys.executable, '-u', str(Path(__file__)), '--model-path', str(args.model_path), *command]
        print('[RUN]', ' '.join(cmd), flush=True)
        with log.open('w', encoding='utf-8') as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(line, end='', flush=True)
            if proc.wait() != 0:
                raise RuntimeError(f'Experiment failed: {log}')
    run('--mode', 'infer', '--tag', 'baseline', '--output', str(result / 'baseline'))
    run('--mode', 'sft', '--steps', '60', '--output', str(result / 'sft'))
    sft = str(result / 'sft' / 'adapter')
    run('--mode', 'eval', '--adapter', sft, '--tag', 'sft', '--output', str(result / 'sft_eval'))
    for version, beta, name in [('original', .1, 'dpo_original_b01'),
                              ('expanded', .1, 'dpo_expanded_b01'),
                              ('expanded', .05, 'dpo_expanded_b005'),
                              ('expanded', .5, 'dpo_expanded_b05')]:
        out = str(result / name)
        run('--mode', 'dpo', '--sft-adapter', sft, '--pairs', version,
            '--beta', str(beta), '--steps', '40', '--output', out)
        run('--mode', 'eval', '--adapter', str(Path(out) / 'adapter'),
            '--tag', name, '--output', str(result / (name + '_eval')))
    for name in ['baseline', 'sft', 'dpo_original_b01', 'dpo_expanded_b01']:
        extra = [] if name == 'baseline' else ['--adapter', str(result / name / 'adapter')]
        run('--mode', 'eval', '--question-set', 'extra', '--tag', name,
            '--output', str(result / (name + '_probe')), *extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['make_data', 'infer', 'sft', 'dpo', 'eval', 'verify', 'all'], default='infer')
    ap.add_argument('--model-path', type=Path, default=ROOT / 'models' / 'SmolVLM-256M-Instruct')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--adapter', type=Path)
    ap.add_argument('--sft-adapter', type=Path, default=ROOT / 'results' / 'sft' / 'adapter')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--beta', type=float, default=.1)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pairs', choices=['original', 'expanded'], default='expanded')
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--max-new-tokens', type=int, default=24)
    ap.add_argument('--question-set', choices=['original', 'extra'], default='original')
    ap.add_argument('--dpo_check')
    args = ap.parse_args()
    if args.dpo_check:
        dpo_check(args.dpo_check)
        return
    seed_all(args.seed)
    if args.steps is not None and args.steps <= 0:
        ap.error('--steps must be positive')
    if args.beta <= 0:
        ap.error('--beta must be positive')
    if args.mode == 'make_data': make_images()
    elif args.mode == 'verify': verify()
    elif args.mode == 'all': run_all(args)
    else:
        if args.output is None:
            args.output = ROOT / 'results' / args.mode
        args.output.mkdir(parents=True, exist_ok=True)
        if args.mode in ['sft', 'dpo']: train(args)
        elif args.mode == 'eval' and args.adapter is None and args.question_set == 'original':
            from copy import copy
            for tag, folder in [('baseline', None), ('sft', 'sft'),
                                ('dpo_expanded_b01', 'dpo_expanded_b01')]:
                stage_args = copy(args)
                stage_args.tag = tag
                stage_args.adapter = ROOT / 'results' / folder / 'adapter' if folder else None
                if tag == 'dpo_expanded_b01' and (ROOT / 'results' / 'dpo' / 'adapter').exists():
                    stage_args.adapter = ROOT / 'results' / 'dpo' / 'adapter'
                if stage_args.adapter and not stage_args.adapter.exists():
                    raise FileNotFoundError(stage_args.adapter)
                stage_args.output = args.output / tag
                stage_args.output.mkdir(parents=True, exist_ok=True)
                evaluate(stage_args)
                gc.collect()
                torch.cuda.empty_cache()
        else: evaluate(args)


if __name__ == '__main__':
    main()
