"""Portable reproduction of the published three-track experiment matrix."""
import argparse
from argparse import Namespace
import gc
from pathlib import Path
import shutil
import torch
import hw6_rsi_localize as h


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finish_seed(args, gt, processor, model):
    qs = h.questions(gt, args.grid)
    _, metrics = h.evaluate(args, processor, model, qs, 'sft')
    h.gate(args, metrics, 'seed')
    h.evaluate(args, processor, model, h.questions(gt, args.grid, 'heldout'), 'sft', 'heldout')
    _, metrics = h.evaluate(args, processor, model, qs, 'harness', correction=True)
    h.gate(args, metrics, 'harness')


def run(args, b1=False):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {args.output}')
    args.output.mkdir(parents=True)
    h.environment(args)
    gt = h.load_gt()
    h.dump(args.output/'questions.json', {
        'eval': h.questions(gt, args.grid), 'heldout': h.questions(gt, args.grid, 'heldout')})
    h.dump(args.output/'protocol.json', dict(
        grid=args.grid, seed_steps=120 if b1 else 10, rsi_steps_per_round=10,
        rsi_rounds=2, supervise_eos=False, predeclared_seed_checkpoint=120 if b1 else 10,
        checkpoint_selection='fixed budget, no independent-test selection',
        fresh_base=True, no_cached_adapter_required=True))
    if not b1:
        processor, model = h.load_model(args)
        h.evaluate(args, processor, model, h.questions(gt, args.grid), 'baseline')
        h.evaluate(args, processor, model, h.questions(gt, args.grid, 'heldout'), 'baseline', 'heldout')
        del processor, model
        release()
    pool = h.make_seed(args, gt)
    processor, model = h.load_model(args, train=True)
    opt = h.optimizer_for(model)
    if b1:
        args.steps = 60
        for cumulative in [60, 120]:
            stage = f'sft{cumulative}'
            h.train(args, processor, model, opt, pool, stage)
            meta = h.read(args.output/stage/'training.json')
            meta.update(cumulative_steps=cumulative, updates_in_this_segment=60)
            h.dump(args.output/stage/'training.json', meta)
            h.evaluate(args, processor, model, h.questions(gt, args.grid), stage)
        shutil.copytree(args.output/'sft120', args.output/'sft')
        args.steps = 10
    else:
        h.train(args, processor, model, opt, pool, 'sft')
    finish_seed(args, gt, processor, model)
    del processor, model, opt
    release()
    h.rsi_loop(args, gt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=['all', 'standard4', 'b1', 'grid8'], default='all')
    parser.add_argument('--model-path', type=Path, default=h.ROOT/'models/SmolVLM-256M-Instruct')
    parser.add_argument('--output-root', type=Path, default=h.ROOT/'results')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--accept-low-accuracy', action='store_true',
                        help='Explicitly accept weak metrics; numerical failures still stop')
    cli = parser.parse_args()
    specs = [('b1', 'b1_grid4', 4, True), ('standard4', 'required_grid4', 4, False),
             ('grid8', 'required_grid8', 8, False)]
    for key, tag, grid, b1 in specs:
        if cli.track not in ['all', key]:
            continue
        args = Namespace(output=cli.output_root/tag, grid=grid, steps=10, rounds=2,
            model_path=cli.model_path, device=cli.device, supervise_eos=False,
            accept_low_accuracy=cli.accept_low_accuracy)
        print(f'=== {tag} ===', flush=True)
        run(args, b1)


if __name__ == '__main__':
    try:
        main()
    except h.QualityPause as exc:
        print('STOP: '+str(exc), flush=True)
        raise SystemExit(20)
