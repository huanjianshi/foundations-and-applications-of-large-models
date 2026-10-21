# -*- coding: utf-8 -*-
"""HW6: reproducible FP32 LoRA, image-only verification and gated RSI.

Use a Python environment with requirements.txt and a local model folder.
Exit code 20 means stop and ask the human; no report or later run is started.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTHONUTF8', '1')
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
COLORS = {'红': (220, 50, 50), '蓝': (50, 80, 220),
          '绿': (40, 170, 70), '黄': (230, 190, 40)}
IMAGE_SIZE = 256
RUN_SEED = 2026


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def seed_all(seed=RUN_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def draw_scene(rnd):
    items = []
    for _ in range(rnd.randint(1, 4)):
        color = rnd.choice(list(COLORS))
        shape = rnd.choice(['圆', '方'])
        size = rnd.randint(34, 54)
        x, y = rnd.randint(8, 256-size-8), rnd.randint(8, 256-size-8)
        items.append([color+shape, [x, y, x+size, y+size]])
    return items


def render(items):
    im = Image.new('RGB', (256, 256), 'white')
    d = ImageDraw.Draw(im)
    for cls, box in items:
        fn = d.ellipse if cls[1] == '圆' else d.rectangle
        fn(box, outline=COLORS[cls[0]], width=4)
    return im


def make_data():
    """Preserve the guide RNG calls, geometry, outlines and question images."""
    gt = {}
    for split, seed, n, prefix in [('train', 2026, 40, 't'), ('eval', 42, 15, 'e'),
                                    ('heldout', 20261020, 30, 'h')]:
        rnd = random.Random(seed)
        folder = ROOT/'hw6_imgs'/split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            key = f'{prefix}{i:02d}'
            items = draw_scene(rnd)
            gt[key] = items
            render(items).save(folder/(key+'.png'))
    dump(ROOT/'hw6_imgs/gt.json', gt)
    manifest = {str(p.relative_to(ROOT)): sha(p)
                for p in sorted((ROOT/'hw6_imgs').glob('*/*.png'))}
    eval_hashes = {v for k, v in manifest.items() if '/eval/' in k.replace('\\', '/')}
    held_hashes = {v for k, v in manifest.items() if '/heldout/' in k.replace('\\', '/')}
    assert not eval_hashes & held_hashes, 'heldout overlaps seed images'
    dump(ROOT/'hw6_imgs/manifest.json', {'seeds': {'train': 2026, 'eval': 42,
         'heldout': 20261020}, 'image_sha256': manifest,
         'heldout_disjoint_from_seed_images': True})
    print('[data] 40 train + 15 guide-eval + 30 independent images', flush=True)
    return gt


def load_gt():
    p = ROOT/'hw6_imgs/gt.json'
    return read(p) if p.exists() else make_data()


def cell_of(box, grid):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    return (max(0, min(grid-1, int(cy*grid/256))),
            max(0, min(grid-1, int(cx*grid/256))))


def cell_box(cell, grid):
    r, c = cell
    return [c*256/grid, r*256/grid, (c+1)*256/grid, (r+1)*256/grid]


def cell_text(cell):
    return f'第{cell[0]+1}行第{cell[1]+1}列'


def parse_cell(text, grid):
    patterns = [r'第\s*(\d+)\s*行\s*第\s*(\d+)\s*列',
                r'\br\s*(\d+)\s*c\s*(\d+)\b',
                r'(?<!\d)(\d+)\s*[,，、\-]\s*(\d+)(?!\d)']
    found = set()
    for pattern in patterns:
        found.update((int(a)-1, int(b)-1) for a, b in re.findall(pattern, text, re.I))
    if len(found) != 1:
        return None
    r, c = next(iter(found))
    return (r, c) if 0 <= r < grid and 0 <= c < grid else None


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.0


def name_of(cls):
    return cls[0]+'色'+cls[1:]+'形'


def question(items, key, index, kind, grid, target=None, ident=None):
    present = sorted({cls for cls, _ in items})
    cls = target or present[index % len(present)]
    q = {'id': ident or f'{kind[0].upper()}{index+1:02d}', 'type': kind,
         'img': key, 'target': cls}
    if kind == 'loc':
        box = next(b for c, b in items if c == cls)
        cell = cell_of(box, grid)
        q.update(q=f'图中的{name_of(cls)}位于第几行第几列？请用「第 X 行第 Y 列」回答（共{grid}行{grid}列）。',
                 gold=cell_text(cell), cell=list(cell), box=box)
    elif kind == 'count':
        n = sum(c == cls for c, _ in items)
        q.update(q=f'图中有几个{name_of(cls)}？', gold=f'{n}个', n=n)
    else:
        exists = any(c == cls for c, _ in items)
        q.update(q=f'图里有没有{name_of(cls)}？', gold='有' if exists else '没有')
    return q


def questions(gt, grid, split='eval'):
    if split == 'eval':
        return [question(gt[f'e{i:02d}'], f'e{i:02d}', i, kind, grid,
                         ident=f'{prefix}{i+1:02d}')
                for kind, prefix, n in [('loc', 'L', 8), ('count', 'C', 4), ('exist', 'E', 3)]
                for i in range(n)]
    qs = []
    all_classes = [c+s for c in COLORS for s in ['圆', '方']]
    for i in range(30):
        key, items = f'h{i:02d}', gt[f'h{i:02d}']
        kind = 'loc' if i < 16 else ('count' if i < 24 else 'exist')
        target = None
        if kind == 'exist' and i % 2:
            present = {c for c, _ in items}
            target = next(c for c in all_classes if c not in present)
        qs.append(question(items, key, i, kind, grid, target, f'H{i+1:02d}'))
    return qs


def image(key):
    split = {'t': 'train', 'e': 'eval', 'h': 'heldout'}[key[0]]
    with Image.open(ROOT/'hw6_imgs'/split/(key+'.png')) as im:
        return im.convert('RGB')


def parse_count(pred):
    low = pred.strip().lower()
    if '?' in low or '？' in low:
        return None
    # Do not split an integral decimal such as "1.0." into the counts 1 and 0.
    values = [Decimal(s) for s in re.findall(r'[+-]?\d+(?:\.\d+)?', low)]
    if any(v < 0 or v != v.to_integral_value() for v in values):
        return None
    nums = {int(v) for v in values}
    for word, n in [('zero', 0), ('one', 1), ('two', 2), ('three', 3), ('four', 4)]:
        if re.search(r'\b'+word+r'\b', low):
            nums.add(n)
    for word, n in [('零', 0), ('一', 1), ('二', 2), ('两', 2), ('三', 3), ('四', 4)]:
        if re.fullmatch(r'(?:答案[是为])?'+word+r'(?:个)?[。.!！]*', low) or word+'个' in low:
            nums.add(n)
    return next(iter(nums)) if len(nums) == 1 else None


def parse_exist(pred):
    low = pred.strip().lower()
    if '?' in low or '？' in low or '有没有' in low:
        return None
    neg = bool(re.search(r'没有|不存在|未见|看不到|\bno\b|\bnot\b|\bnone\b', low)
               or re.fullmatch(r'[否无不][。.!！]*', low))
    explicit_yes = bool(re.search(r'\byes\b', low))
    pos = bool(re.search(r'有|存在|there (?:is|are)|\byes\b', low))
    if neg and explicit_yes:
        return None
    return False if neg else (True if pos else None)


def score(q, pred, grid):
    """GT is confined to scoring, not the image verifier or control flow."""
    if q['type'] == 'loc':
        c = parse_cell(pred, grid)
        return {'correct': int(c == tuple(q['cell'])), 'format_valid': int(c is not None),
                'iou': iou(cell_box(c, grid), q['box']) if c is not None else 0.0}
    if q['type'] == 'count':
        n = parse_count(pred)
        return {'correct': int(n == q['n']), 'format_valid': int(n is not None), 'iou': None}
    x = parse_exist(pred)
    return {'correct': int(x is not None and x == (q['gold'] == '有')),
            'format_valid': int(x is not None), 'iou': None}


def guide_judge(q, pred, grid):
    """Also retain the reference's more permissive literal scoring for audit."""
    low = pred.strip().lower()
    if q['type'] == 'loc':
        return int(parse_cell(pred, grid) == tuple(q['cell']))
    if q['type'] == 'count':
        ns = re.findall(r'\d+', low)
        return int(bool(ns) and int(ns[0]) == q['n'])
    cleaned = low.replace('有没有', '')
    yes = ('有' in low and '没有' not in cleaned) or 'yes' in low
    no = '没有' in cleaned or 'no' in low or 'not' in low
    return int(yes if q['gold'] == '有' else no)


def color_components(im, color):
    im = im.convert('RGB')
    px, ref = im.load(), COLORS[color]
    pts = {(x, y) for y in range(im.height) for x in range(im.width)
           if sum((px[x, y][j]-ref[j])**2 for j in range(3)) < 9000}
    comps = []
    while pts:
        start = pts.pop()
        stack, component = [start], [start]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    v = (x+dx, y+dy)
                    if v in pts:
                        pts.remove(v)
                        stack.append(v)
                        component.append(v)
        xs, ys = zip(*component)
        comps.append(([min(xs), min(ys), max(xs)+1, max(ys)+1], len(component)))
    return sorted(comps)


def verify(im, target, kind, pred, grid):
    """Image and target only: deliberately no q.gold, q.cell, q.box or GT."""
    comps = color_components(im, target[0])
    evidence = {'color': target[0], 'components': len(comps), 'boxes': comps,
                'pixels': sum(a for _, a in comps), 'shape_aware': False}
    if kind == 'loc':
        c = parse_cell(pred, grid)
        if c is None:
            return False, '输出不是有效网格坐标，请按指定格式重新作答。', evidence
        cb = cell_box(c, grid)
        overlap = 0.0
        for box, _ in comps:
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            if cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]:
                return True, '该颜色连通块的中心在高亮格内。', evidence
            inter = max(0, min(cb[2], box[2])-max(cb[0], box[0])) * max(0, min(cb[3], box[3])-max(cb[1], box[1]))
            overlap = max(overlap, inter/((box[2]-box[0])*(box[3]-box[1])))
        evidence['best_overlap'] = overlap
        passed = overlap >= .5
        msg = f'预测格覆盖该颜色连通块的最大面积比例为{overlap:.0%}，'+('通过复核。' if passed else '请重新观察位置并给出网格坐标。')
        return passed, msg, evidence
    if kind == 'count':
        return parse_count(pred) == len(comps), f'颜色分析发现该颜色有{len(comps)}个连通块，请核对数量。', evidence
    present = bool(comps)
    return parse_exist(pred) == present, f'颜色分析发现该颜色像素数为{evidence["pixels"]}，请核对存在性。', evidence


def overlay(im, c, grid):
    im = im.copy()
    d = ImageDraw.Draw(im)
    box = cell_box(c, grid)
    d.rectangle(box, outline=(255, 0, 255), width=5)
    d.rectangle([box[0]+6, box[1]+6, box[2]-6, box[3]-6], outline=(255, 0, 255), width=2)
    return im


def build_prompt(processor, q):
    return processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'image'}, {'type': 'text', 'text': q}]}], add_generation_prompt=True)


def encode_answer(processor, im, prompt, response, supervise_eos=False):
    text = build_prompt(processor, prompt)
    prefix = processor(text=[text], images=[im], return_tensors='pt')
    suffix = response + (processor.tokenizer.eos_token if supervise_eos else '')
    full = processor(text=[text+suffix], images=[im], return_tensors='pt')
    plen = prefix['input_ids'].shape[1]
    assert torch.equal(prefix['input_ids'], full['input_ids'][:, :plen]), 'changed token boundary'
    labels = full['input_ids'].clone()
    labels[:, :plen] = -100
    labels[full['attention_mask'] == 0] = -100
    assert labels[:, plen:].ne(-100).any(), 'no supervised answer token'
    assert labels[:, :plen].eq(-100).all(), 'prompt must be masked'
    full['labels'] = labels
    return full, {'prompt_tokens': plen, 'answer_tokens': int(labels.ne(-100).sum()),
                  'supervised_text': processor.decode(labels[labels.ne(-100)]),
                  'supervise_eos': supervise_eos}


def load_model(args, train=False, adapter=None):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, PeftModel, get_peft_model
    seed_all()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.image_processor.size = {'longest_edge': 512}
    processor.image_processor.do_image_splitting = False
    base = AutoModelForVision2Seq.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa').to(args.device)
    if adapter:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=train)
    elif train:
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    else:
        model = base
    model.eval()
    return processor, model


@torch.no_grad()
def answer(processor, model, im, q, max_new=24):
    model.eval()
    device = next(model.parameters()).device
    batch = processor(text=[build_prompt(processor, q)], images=[im], return_tensors='pt').to(device)
    sync(device)
    start = time.perf_counter()
    result = model.generate(**batch, max_new_tokens=max_new, do_sample=False, use_cache=True)
    sync(device)
    generated = result[0, batch['input_ids'].shape[1]:]
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    ended = bool(len(generated) and int(generated[-1]) in eos)
    return {'prediction': processor.decode(generated, skip_special_tokens=True).strip(),
            'generated_tokens': len(generated), 'ended_by_eos': int(ended),
            'reached_token_limit': int(len(generated) >= max_new and not ended),
            'seconds': time.perf_counter()-start}


def summarize(rows, stage, split):
    result = []
    for kind in ['loc', 'count', 'exist', 'total']:
        subset = rows if kind == 'total' else [r for r in rows if r['type'] == kind]
        loc = [r for r in subset if r['type'] == 'loc']
        result.append({'stage': stage, 'split': split, 'type': kind, 'n': len(subset),
            'correct': sum(r['correct'] for r in subset),
            'accuracy': sum(r['correct'] for r in subset)/len(subset),
            'guide_correct': sum(r['guide_correct'] for r in subset),
            'format_valid': sum(r['format_valid'] for r in subset),
            'mean_iou': sum(r['iou'] for r in loc)/len(loc) if loc else None,
            'token_limit_count': sum(r['reached_token_limit'] for r in subset)})
    return result


def evaluate(args, processor, model, qs, stage, split='eval', correction=False):
    rows, traces = [], []
    for q in qs:
        im = image(q['img'])
        generated = answer(processor, model, im, q['q'])
        if correction:
            generated, trace = correct(args, processor, model, q, generated, stage, self_check=False)
            traces.append(trace)
        pred = generated['prediction']
        row = dict(stage=stage, split=split, id=q['id'], type=q['type'], img=q['img'],
                   target=q['target'], question=q['q'], gold=q['gold'], **generated,
                   **score(q, pred, args.grid), guide_correct=guide_judge(q, pred, args.grid))
        rows.append(row)
        print(f'[{stage}/{split}/{q["id"]}] {row["correct"]} {pred!r} gold={q["gold"]}', flush=True)
    metrics = summarize(rows, stage, split)
    dump(args.output/stage/(split+'_predictions.json'), rows)
    csv_write(args.output/stage/(split+'_predictions.csv'), rows)
    dump(args.output/stage/(split+'_summary.json'), metrics)
    if traces:
        dump(args.output/stage/'correction_traces.json', traces)
    all_rows = []
    for p in sorted(args.output.glob('*/*_summary.json')):
        all_rows.extend(read(p))
    csv_write(args.output/'results.csv', all_rows)
    print('[summary]', json.dumps(metrics, ensure_ascii=False), flush=True)
    return rows, metrics


def correct(args, processor, model, q, original, stage, self_check=True):
    im = image(q['img'])
    current = original
    trace = {'id': q['id'], 'img': q['img'], 'type': q['type'], 'target': q['target'],
             'question': q['q'], 'original': original, 'self_called': False, 'tool_called': False}
    c = parse_cell(current['prediction'], args.grid) if q['type'] == 'loc' else None
    if self_check and c is not None:
        marked = overlay(im, c, args.grid)
        p = args.output/stage/'overlays'/(q['id']+'.png')
        p.parent.mkdir(parents=True, exist_ok=True)
        marked.save(p)
        current = answer(processor, model, marked,
            '图中紫色高亮框是上一次给出的位置。'+q['q']+'\n若高亮框没有圈住目标，请直接给出正确位置。', 16)
        trace.update(self_called=True, self_answer=current,
                     self_changed=current['prediction'] != original['prediction'])
    passed, msg, evidence = verify(im, q['target'], q['type'], current['prediction'], args.grid)
    trace.update(pre_tool_passed=passed, pre_tool_evidence=evidence)
    if not passed:
        current = answer(processor, model, im, q['q']+'\n（校验提示：'+msg+'）')
        trace.update(tool_called=True, hint=msg, tool_answer=current)
    passed, msg, evidence = verify(im, q['target'], q['type'], current['prediction'], args.grid)
    trace.update(final=current, verifier_passed=passed, final_evidence=evidence)
    # Only after all decisions have been made, attach GT scores for analysis.
    trace['original_score'] = score(q, original['prediction'], args.grid)
    trace['self_score'] = score(q, trace['self_answer']['prediction'], args.grid) if trace['self_called'] else None
    trace['final_score'] = score(q, current['prediction'], args.grid)
    return current, trace


def make_seed(args, gt):
    pool = [dict(img=q['img'], prompt=q['q'], answer=q['gold'], source='GT seed',
                 question_id=q['id']) for q in questions(gt, args.grid)]
    dump(args.output/'seed_data.json', pool)
    dump(args.output/'sft_data.json', pool)
    if args.grid == 4:
        dump(ROOT/'sft_data.json', pool)
    return pool


def train(args, processor, model, optimizer, pool, stage, round_idx=0, new_data=None):
    assert pool and all(s['img'].startswith('e') for s in pool), 'heldout cannot be trained'
    rnd = random.Random(RUN_SEED+round_idx)
    if round_idx == 0:
        selected = [pool[i % len(pool)] for i in range(args.steps)]  # guide order
    else:
        new_data = new_data or []
        n = min(5, len(new_data), args.steps)
        selected = rnd.sample(new_data, n) + [rnd.choice(pool) for _ in range(args.steps-n)]
        rnd.shuffle(selected)
    cache, audits = {}, []
    for s in selected:
        key = (s['img'], s['prompt'], s['answer'])
        if key not in cache:
            cache[key], audit = encode_answer(processor, image(s['img']), s['prompt'], s['answer'],
                                             getattr(args, 'supervise_eos', False))
            audits.append(dict(img=s['img'], prompt=s['prompt'], answer=s['answer'], **audit))
    dump(args.output/stage/'mask_check.json', audits)
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    start = time.perf_counter()
    history = []
    for i, s in enumerate(selected):
        batch = cache[(s['img'], s['prompt'], s['answer'])].to(device)
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            dump(args.output/'quality_gate.json', {'status': 'PAUSED', 'reason': 'nonfinite loss', 'stage': stage})
            raise QualityPause('nonfinite training loss')
        loss.backward()
        norm = math.sqrt(sum(float(p.grad.detach().float().norm())**2
                             for p in model.parameters() if p.grad is not None))
        if not math.isfinite(norm):
            dump(args.output/'quality_gate.json', {'status': 'PAUSED', 'reason': 'nonfinite gradient', 'stage': stage})
            raise QualityPause('nonfinite gradient')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        row = dict(step=i+1, img=s['img'], question_id=s['question_id'], source=s['source'],
                   answer=s['answer'], loss=float(loss.detach()), grad_norm=norm,
                   elapsed_seconds=time.perf_counter()-start)
        history.append(row)
        csv_write(args.output/stage/'history.csv', history)
        print(f'[train/{stage}] {i+1}/{args.steps} loss={row["loss"]:.5f} sample={s["question_id"]}', flush=True)
    model.eval()
    folder = args.output/stage/'adapter'
    model.save_pretrained(folder)
    torch.save(optimizer.state_dict(), args.output/stage/'optimizer.pt')
    full_pool = pool + (new_data or []) if round_idx else pool
    dump(args.output/stage/'pool.json', full_pool)
    dump(args.output/stage/'training.json', dict(steps=args.steps, seconds=time.perf_counter()-start,
        pool_entries=len(full_pool), unique_image_questions=len({(s['img'], s['prompt']) for s in full_pool}),
        new_sample_steps=sum(s['source'] == f'RSI round {round_idx}' for s in selected),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total=sum(p.numel() for p in model.parameters()), precision='FP32 LoRA',
        supervise_eos=getattr(args, 'supervise_eos', False)))
    # Keep these CPU tensors from accumulating on GPU across rounds.
    del cache
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()


class QualityPause(RuntimeError):
    pass


def gate(args, metrics, checkpoint):
    m = {r['type']: r for r in metrics}
    reasons = []
    if checkpoint == 'seed':
        if m['total']['correct'] < 8:
            reasons.append('seed total below 8/15')
        if m['loc']['correct'] < 4:
            reasons.append('seed localization below 4/8')
    elif checkpoint in ['harness', 'final'] and args.grid == 4:
        if m['total']['correct'] < 12:
            reasons.append('4x4 total below 12/15')
        if m['loc']['correct'] < 6:
            reasons.append('4x4 localization below 6/8')
    elif checkpoint == 'heldout' and args.grid == 4:
        if m['total']['correct'] < 21:
            reasons.append('heldout total below 21/30')
    elif checkpoint == 'final' and args.grid == 8:
        if m['total']['accuracy'] < .5 or m['loc']['accuracy'] < .5:
            reasons.append('8x8 majority of total or localization predictions wrong')
    # Baseline is never gated. For trained models, >=50% invalid localization is abnormal.
    if checkpoint != 'heldout' and m['loc']['format_valid'] <= m['loc']['n']/2:
        reasons.append('at least half of localization outputs are unparseable')
    accepted = bool(reasons and getattr(args, 'accept_low_accuracy', False))
    status = {'status': ('ACCEPTED_LOW_ACCURACY' if accepted else 'PAUSED') if reasons else 'PASSED', 'checkpoint': checkpoint,
              'grid': args.grid, 'reasons': reasons, 'metrics': metrics,
              'user_authorization': 'Explicit --accept-low-accuracy flag' if accepted else None,
              'next_action': 'continue and report limitations' if accepted else ('Ask user; no further experiments or report.' if reasons else 'continue')}
    dump(args.output/f'gate_{checkpoint}.json', status)
    dump(args.output/'quality_gate.json', status)
    if reasons and not accepted:
        print('[QUALITY PAUSE]', json.dumps(status, ensure_ascii=False), flush=True)
        raise QualityPause('; '.join(reasons))


def environment(args):
    versions = {}
    for p in ['torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'pillow', 'bitsandbytes']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = 'not installed'
    sources = {p.name: sha(p) for p in ROOT.iterdir()
               if p.suffix in ['.docx', '.pdf'] and ('样板' in p.name or '指南' in p.name)}
    dump(args.output/'environment.json', dict(python=sys.version, executable=sys.executable,
        versions=versions, device=args.device, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        grid=args.grid, model=str(args.model_path), model_sha256=sha(args.model_path/'model.safetensors'),
        source_material_sha256=sources, code_sha256=sha(__file__),
        processor_longest_edge=512, image_splitting=False, precision='FP32',
        seed=RUN_SEED, sft_steps=args.steps, learning_rate=1e-4, lora_r=8, lora_alpha=16,
        lora_dropout=.05, seed_train_evaluation_overlap=True,
        independent_test_used_for_training_or_selection=False,
        supervise_eos=getattr(args, 'supervise_eos', False)))
    (args.output/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def optimizer_for(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)


def rsi_loop(args, gt):
    existing = args.output/'quality_gate.json'
    if existing.exists() and read(existing)['status'] == 'PAUSED' and not getattr(args, 'accept_low_accuracy', False):
        raise QualityPause('Existing quality pause requires an explicit human decision before resuming.')
    adapter = args.output/'sft/adapter'
    if not adapter.exists():
        raise FileNotFoundError('Run seed SFT and its quality gate first.')
    processor, model = load_model(args, train=True, adapter=adapter)
    opt = optimizer_for(model)
    opt.load_state_dict(torch.load(args.output/'sft/optimizer.pt', map_location=args.device, weights_only=True))
    pool = read(args.output/'seed_data.json')
    mem, rounds = [], []
    qs = questions(gt, args.grid)
    for r in range(1, args.rounds+1):
        stage = f'rsi_round{r}'
        traces, new_data = [], []
        for q in qs:
            original = answer(processor, model, image(q['img']), q['q'])
            final, trace = correct(args, processor, model, q, original, stage)
            trace['round'] = r
            traces.append(trace)
            if trace['verifier_passed']:
                new_data.append(dict(img=q['img'], prompt=q['q'], answer=final['prediction'],
                    question_id=q['id'], source=f'RSI round {r}'))
        mem.extend(traces)
        dump(args.output/'rsi_memory.json', mem)
        if args.grid == 4:
            dump(ROOT/'rsi_memory.json', mem)
        old_pool = pool[:]
        pool += new_data
        dump(args.output/'sft_data.json', pool)
        if args.grid == 4:
            dump(ROOT/'sft_data.json', pool)
        # Old pool is used for the replay portion; fresh samples have guaranteed exposure.
        train(args, processor, model, opt, old_pool, stage, r, new_data)
        dump(args.output/stage/'pool.json', pool)
        rows, metrics = evaluate(args, processor, model, qs, stage)
        record = dict(round=r, pool_entries=len(pool), unique_image_questions=len({(s['img'], s['prompt']) for s in pool}),
            direct_correct=sum(t['original_score']['correct'] for t in traces),
            self_calls=sum(t['self_called'] for t in traces),
            self_changes=sum(t.get('self_changed', False) for t in traces),
            tool_calls=sum(t['tool_called'] for t in traces),
            final_correct=sum(t['final_score']['correct'] for t in traces), new_entries=len(new_data),
            verifier_false_accepts=sum(t['verifier_passed'] and not t['final_score']['correct'] for t in traces),
            wrong_to_right=sum(not t['original_score']['correct'] and t['final_score']['correct'] for t in traces),
            right_to_wrong=sum(t['original_score']['correct'] and not t['final_score']['correct'] for t in traces),
            end_accuracy=next(m['accuracy'] for m in metrics if m['type'] == 'total'))
        rounds.append(record)
        csv_write(args.output/'rounds.csv', rounds)
        if r == args.rounds:
            gate(args, metrics, 'final')
        held_rows, held_metrics = evaluate(args, processor, model, questions(gt, args.grid, 'heldout'), stage, 'heldout')
        if r == args.rounds:
            gate(args, held_metrics, 'heldout')
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['make_data', 'make_sft', 'make_seed', 'infer', 'sft', 'eval', 'rsi', 'all'], default='infer')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/SmolVLM-256M-Instruct')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--grid', type=int, choices=[4, 8], default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--supervise-eos', action='store_true', help='Explicit extension; original default remains no EOS')
    parser.add_argument('--accept-low-accuracy', action='store_true', help='Only with explicit human authorization; numerical failures still stop')
    args = parser.parse_args()
    args.output = args.output or ROOT/'results'/f'grid{args.grid}'
    if args.steps < 1 or args.rounds < 1:
        parser.error('steps and rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'make_data':
        make_data()
        return
    gt = load_gt()
    qs = questions(gt, args.grid)
    dump(args.output/'questions.json', {'eval': qs, 'heldout': questions(gt, args.grid, 'heldout')})
    if args.mode in ['make_seed', 'make_sft']:
        make_seed(args, gt)
        return
    environment(args)
    if args.mode == 'rsi':
        rsi_loop(args, gt)
        return
    if args.mode in ['infer', 'all']:
        processor, model = load_model(args)
        evaluate(args, processor, model, qs, 'baseline')
        if args.mode == 'infer':
            return
        evaluate(args, processor, model, questions(gt, args.grid, 'heldout'), 'baseline', 'heldout')
        del model, processor
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if args.mode in ['sft', 'all']:
        if (args.output/'sft/adapter').exists():
            raise FileExistsError('Refusing to overwrite existing SFT. Choose a new --output.')
        pool = make_seed(args, gt)
        processor, model = load_model(args, train=True)
        train(args, processor, model, optimizer_for(model), pool, 'sft')
        rows, metrics = evaluate(args, processor, model, qs, 'sft')
        gate(args, metrics, 'seed')
        evaluate(args, processor, model, questions(gt, args.grid, 'heldout'), 'sft', 'heldout')
        if args.mode == 'sft':
            return
        _, metrics = evaluate(args, processor, model, qs, 'harness', correction=True)
        gate(args, metrics, 'harness')
        del model, processor
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        rsi_loop(args, gt)
    elif args.mode == 'eval':
        processor, model = load_model(args)
        evaluate(args, processor, model, qs, 'baseline')
        del model
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        processor, model = load_model(args, adapter=args.output/'sft/adapter')
        _, metrics = evaluate(args, processor, model, qs, 'sft')
        gate(args, metrics, 'seed')
        _, metrics = evaluate(args, processor, model, qs, 'harness', correction=True)
        gate(args, metrics, 'harness')


if __name__ == '__main__':
    try:
        main()
    except QualityPause as exc:
        print('STOP: '+str(exc), flush=True)
        sys.exit(20)
