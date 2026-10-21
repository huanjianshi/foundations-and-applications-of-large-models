"""Authorized path A only: fixed 10-step EOS/prompt factorial diagnostics.

Never changes or resumes the original paused run. Never reads heldout answers.
The control cell is the saved original run; three other cells start from base.
"""
from argparse import Namespace
import argparse
from pathlib import Path
import gc
import json
import shutil
import time

import torch
import hw6_rsi_localize as h

BASE = h.ROOT/'diagnostics/format_A'


def compact(q):
    target = h.name_of(q['target'])
    if q['type'] == 'loc':
        return (f'将图片等分为4行4列，行从上到下、列从左到右编号1到4。'
                f'{target}的中心在哪格？只输出坐标，格式：第r行第c列。')
    return q['q']


def metrics_entry(tag, metrics, rows):
    m = {r['type']: r for r in metrics}
    return dict(arm=tag, loc_correct=m['loc']['correct'], loc_n=m['loc']['n'],
        count_correct=m['count']['correct'], exist_correct=m['exist']['correct'],
        total_correct=m['total']['correct'], total_n=m['total']['n'],
        loc_format_valid=m['loc']['format_valid'], mean_iou=m['loc']['mean_iou'],
        token_limit_count=m['total']['token_limit_count'],
        guide_correct=m['total']['guide_correct'],
        seed_threshold_passed=m['total']['correct'] >= 8 and m['loc']['correct'] >= 4,
        prediction_source='saved original (rescored)' if tag == 'guide_noeos' else 'fresh 10-step arm')


def audit_saved_control():
    gt = h.load_gt()
    qs = {q['id']: q for q in h.questions(gt, 4)}
    summaries = []
    for stage in ['baseline', 'sft']:
        rows = h.read(h.ROOT/f'artifacts/required_grid4/{stage}_eval_predictions.json')
        revised = []
        for old in rows:
            q = qs[old['id']]
            revised.append(dict(old, **h.score(q, old['prediction'], 4)))
        metrics = h.summarize(revised, stage, 'eval')
        h.dump(BASE/'rescored_original'/f'{stage}_predictions.json', revised)
        h.csv_write(BASE/'rescored_original'/f'{stage}_predictions.csv', revised)
        h.dump(BASE/'rescored_original'/f'{stage}_summary.json', metrics)
        if stage == 'sft':
            summaries.append(metrics_entry('guide_noeos', metrics, revised))
    return summaries


def run_arm(tag, eos, short_prompt):
    out = BASE/tag
    if (out/'sft/adapter').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    args = Namespace(model_path=MODEL_PATH,
                     output=out, device='cuda', grid=4, steps=10, rounds=2, supervise_eos=eos)
    h.environment(args)
    (out/'diagnostic_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    qs = h.questions(h.load_gt(), 4)
    original_questions = [dict(q) for q in qs]
    if short_prompt:
        for q in qs:
            q['q'] = compact(q)
    pool = [dict(img=q['img'], prompt=q['q'], answer=q['gold'], source='GT seed',
                 question_id=q['id']) for q in qs]
    h.dump(out/'seed_data.json', pool)
    h.dump(out/'config.json', dict(factorial_cell=tag, steps=10, supervise_eos=eos,
        prompt_style='compact explicit grid instruction' if short_prompt else 'guide',
        seed=h.RUN_SEED, sample_order=[s['question_id'] for s in pool[:10]],
        additional_training_steps_beyond_10=0, heldout_used=False,
        other_settings='same model, images, labels, sample order, LoRA, lr, precision and generation budget'))
    processor, model = h.load_model(args, train=True)
    # Inspect every seed's exact labels and verify EOS is confined to the suffix.
    audits = []
    for s in pool:
        batch, audit = h.encode_answer(processor, h.image(s['img']), s['prompt'], s['answer'], eos)
        labels = batch['labels'][batch['labels'].ne(-100)]
        audit['last_label_is_eos'] = int(labels[-1]) == processor.tokenizer.eos_token_id
        assert audit['last_label_is_eos'] == eos
        audits.append(dict(id=s['question_id'], **audit))
    h.dump(out/'all_seed_mask_audit.json', audits)
    opt = h.optimizer_for(model)
    h.train(args, processor, model, opt, pool, 'sft')
    rows, metrics = h.evaluate(args, processor, model, qs, 'sft')
    # A paired generation under original prompts isolates inference prompt effects
    # on the very same compact-trained weights; never trains on test images.
    if short_prompt:
        h.evaluate(args, processor, model, original_questions, 'original_prompt_probe')
    # Read-only format diagnosis: does a larger generation limit fix a truncation?
    probes = []
    for q in qs[:2]:
        generated = h.answer(processor, model, h.image(q['img']), q['q'], max_new=64)
        probes.append(dict(id=q['id'], **generated, **h.score(q, generated['prediction'], 4)))
    h.dump(out/'token_budget_probe.json', probes)
    # Verify saved/reloaded adapter outputs, without updating any weights.
    before = [r['prediction'] for r in rows[:2]]
    del opt, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    processor, reloaded = h.load_model(args, adapter=out/'sft/adapter')
    after = [h.answer(processor, reloaded, h.image(q['img']), q['q'])['prediction'] for q in qs[:2]]
    h.dump(out/'roundtrip_check.json', {'ids': [q['id'] for q in qs[:2]],
        'before': before, 'after': after, 'equal': before == after})
    assert before == after, 'saved adapter changes greedy answers'
    del processor, reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return metrics_entry(tag, metrics, rows)


def main():
    started = time.perf_counter()
    BASE.mkdir(parents=True, exist_ok=True)
    paused_gate = h.ROOT/'results/grid4/quality_gate.json'
    gate_hash_before = h.sha(paused_gate) if paused_gate.exists() else None
    summaries = audit_saved_control()
    h.csv_write(BASE/'comparison.csv', summaries)
    for tag, eos, short_prompt in [('guide_eos', True, False),
                                  ('compact_noeos', False, True),
                                  ('compact_eos', True, True)]:
        summaries.append(run_arm(tag, eos, short_prompt))
        h.csv_write(BASE/'comparison.csv', summaries)
    assert gate_hash_before is None or h.sha(paused_gate) == gate_hash_before, 'original pause must remain intact'
    h.dump(BASE/'diagnosis_status.json', dict(status='A_COMPLETE_AWAITING_DECISION',
        original_control_predictions_preserved=True, report_started=False, rsi_started=False,
        independent_test_used=False, seconds=time.perf_counter()-started, comparison=summaries))
    print('[A COMPLETE]', json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=h.ROOT/'models/SmolVLM-256M-Instruct')
    MODEL_PATH = parser.parse_args().model_path
    main()
