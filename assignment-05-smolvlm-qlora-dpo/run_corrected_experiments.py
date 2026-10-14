"""Reproduce required experiments and declared extensions with real checkpoints.

python run_corrected_experiments.py
python run_corrected_experiments.py --resume  # keep completed training; rerun evaluations
Held-out questions are never used for training or checkpoint selection.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results_corrected'
MODEL = ROOT / 'models' / 'SmolVLM-256M-Instruct'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model-path', type=Path, default=MODEL)
    args = parser.parse_args()
    R.mkdir(exist_ok=True)

    def run(name, *command, train=False):
        out = R if name == 'suite' else R / name
        if train and args.resume and (out / 'adapter' / 'adapter_model.safetensors').exists():
            print('[resume trained checkpoint]', name, flush=True)
            return
        cmd = [sys.executable, '-u', str(ROOT/'hw5_dpo.py'), '--model-path', str(args.model_path),
               '--output', str(out), *map(str, command)]
        print('[RUN]', name, flush=True)
        with (R/(name+'.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{name} failed: {R/(name+".log")}')
        if train:
            (out/'training_source_snapshot.py').write_bytes((ROOT/'hw5_dpo.py').read_bytes())

    if not args.resume or not (R/'dpo_expanded_b05'/'adapter').exists():
        run('suite', '--mode', 'all')
    # Rescore baselines using the final target-aware semantic rule.
    run('baseline', '--mode', 'infer', '--tag', 'baseline', '--eval-precision', 'fp32')
    run('baseline_nf4', '--mode', 'infer', '--tag', 'baseline_nf4', '--eval-precision', 'nf4')
    run('sft120', '--mode', 'sft', '--steps', 120, '--roundtrip-check', train=True)
    run('sft120_eos', '--mode', 'sft', '--steps', 120, '--supervise-eos', '--roundtrip-check', train=True)
    for name, initialization, eos in [
        ('dpo60_lr1e5', 'sft', False), ('dpo120_lr1e5', 'sft120', False),
        ('dpo120_eos_lr1e5', 'sft120_eos', True)]:
        flags = ['--supervise-eos'] if eos else []
        run(name, '--mode', 'dpo', '--steps', 40, '--learning-rate', 1e-5,
            '--checkpoint-every', 10, '--roundtrip-check', '--pairs', 'expanded',
            '--sft-adapter', R/initialization/'adapter', *flags, train=True)
    stages = ['sft', 'dpo_original_b01', 'dpo_expanded_b01', 'dpo_expanded_b005',
              'dpo_expanded_b05', 'sft120', 'sft120_eos', 'dpo60_lr1e5',
              'dpo120_lr1e5', 'dpo120_eos_lr1e5']
    for stage in stages:
        run(stage+'_eval', '--mode', 'eval', '--adapter', R/stage/'adapter', '--tag', stage)
    for stage in ['baseline', 'sft', 'dpo_original_b01', 'dpo_expanded_b01', 'dpo120_eos_lr1e5']:
        flags = [] if stage=='baseline' else ['--adapter', R/stage/'adapter']
        run(stage+'_probe', '--mode', 'eval', '--tag', stage, '--question-set', 'extra', *flags)
    # Inspect original-question generation at each DPO checkpoint. Keep the declared
    # 40-step endpoint as the final model, rather than selecting on held-out results.
    for step in [10, 20, 30, 40]:
        run(f'dpo120_eos_step{step}', '--mode', 'eval', '--tag', f'dpo120_eos_step{step}',
            '--adapter', R/'dpo120_eos_lr1e5'/'checkpoints'/f'step_{step:03d}')
    run('sft_fp32_control', '--mode', 'eval', '--adapter', R/'sft'/'adapter',
        '--eval-precision', 'fp32', '--tag', 'sft_fp32_control')
    run('dpo_fp32_control', '--mode', 'eval', '--adapter', R/'dpo_expanded_b01'/'adapter',
        '--eval-precision', 'fp32', '--tag', 'dpo_fp32_control')
    # Configuration/model choices are frozen before any independent-test evaluation.
    frozen = dict(final_stage='dpo120_eos_lr1e5', sft_steps=120, dpo_steps=40,
                  eos_supervision=True, dpo_learning_rate=1e-5, beta=.1,
                  selection_data='original training-overlap questions only',
                  heldout_used_for_selection=False)
    (R/'final_selection.json').write_text(json.dumps(frozen, indent=2)+'\n', encoding='utf-8')
    for stage in ['baseline', 'baseline_nf4', 'sft', 'dpo_expanded_b01',
                  'sft120_eos', 'dpo120_eos_lr1e5']:
        flags = ['--eval-precision', 'nf4'] if stage=='baseline_nf4' else []
        if stage not in ['baseline', 'baseline_nf4']:
            flags += ['--adapter', R/stage/'adapter']
        run(stage+'_heldout', '--mode', 'eval', '--tag', stage, '--question-set', 'heldout', *flags)
    print('[complete] corrected experiments, extensions, precision controls and independent test', flush=True)


if __name__ == '__main__':
    main()
