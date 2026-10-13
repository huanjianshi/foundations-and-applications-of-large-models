"""Summarize live runs or the checked artifacts; validate and redraw curves.

python summarize_results.py --input-dir results --output-dir results
python summarize_results.py --input-dir artifacts --output-dir reproduced
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hw5_dpo import metrics

STAGES = ['baseline', 'sft', 'dpo_original_b01', 'dpo_expanded_b01',
          'dpo_expanded_b005', 'dpo_expanded_b05']
BETAS = {'dpo_original_b01': .1, 'dpo_expanded_b01': .1,
         'dpo_expanded_b005': .05, 'dpo_expanded_b05': .5}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    if (args.input_dir / 'baseline' / 'predictions.csv').exists():
        predictions = []
        for stage in STAGES:
            folder = 'baseline' if stage == 'baseline' else 'sft_eval' if stage == 'sft' else stage + '_eval'
            predictions.extend(read_csv(args.input_dir / folder / 'predictions.csv'))
    else:
        predictions = read_csv(args.input_dir / 'all_predictions.csv')
    checked = []
    for row in predictions:
        guide, fact = metrics(row, row['prediction'])
        checked.append(dict(stage=row['stage'], id=row['id'], type=row['type'],
            img=row['img'], q=row['q'], gold=row['gold'], prediction=row['prediction'],
            guide_correct=guide, fact_correct=fact))
    summaries = []
    for stage in STAGES:
        rows = [row for row in checked if row['stage'] == stage]
        if len(rows) != 15:
            raise ValueError(f'{stage}: expected 15 outputs, got {len(rows)}')
        for kind in ['count', 'attr', 'exist', 'total']:
            group = rows if kind == 'total' else [row for row in rows if row['type'] == kind]
            summaries.append(dict(stage=stage, type=kind, n=len(group),
                guide_correct=sum(row['guide_correct'] for row in group),
                fact_correct=sum(row['fact_correct'] for row in group)))
    write_csv(out / 'results.csv', summaries)
    write_csv(out / 'all_predictions.csv', checked)
    histories = {}
    stats = {}
    for stage in STAGES[1:]:
        live = args.input_dir / stage / 'history.csv'
        rows = read_csv(live if live.exists() else args.input_dir / ('history_' + stage + '.csv'))
        for row in rows:
            for key in row:
                row[key] = int(row[key]) if key in ['step', 'epoch', 'sample_index'] else float(row[key])
        expected = 60 if stage == 'sft' else 40
        assert [row['step'] for row in rows] == list(range(1, expected + 1))
        assert all(math.isfinite(row['loss']) and row['grad_norm'] > 0 for row in rows)
        if stage in BETAS:
            for row in rows:
                margin = BETAS[stage] * ((row['pol_w'] - row['ref_w']) - (row['pol_l'] - row['ref_l']))
                loss = max(-margin, 0) + math.log1p(math.exp(-abs(margin)))
                assert abs(margin - row['margin']) < 1e-5
                assert abs(loss - row['loss']) < 1e-5
        histories[stage] = rows
        stats[stage] = dict(first_loss=rows[0]['loss'], last_loss=rows[-1]['loss'],
            first10_loss=statistics.mean(row['loss'] for row in rows[:10]),
            last10_loss=statistics.mean(row['loss'] for row in rows[-10:]))
    (out / 'training_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    rows = histories['sft']
    ax.plot([row['step'] for row in rows], [row['loss'] for row in rows], color='#31688e')
    ax.set(xlabel='Optimizer step', ylabel='SFT loss', title='SFT: answer-token supervision, 60 steps')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures / 'sft_loss.png', dpi=220)
    plt.close(fig)
    for filename, stages, labels in [
        ('dpo_loss_margin.png', ['dpo_original_b01', 'dpo_expanded_b01'], ['12 pairs', '17 pairs']),
        ('beta_scan.png', ['dpo_expanded_b005', 'dpo_expanded_b01', 'dpo_expanded_b05'],
         ['beta=0.05', 'beta=0.1', 'beta=0.5'])]:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for stage, label, color in zip(stages, labels, ['#31688e', '#35b779', '#d95f02']):
            rows = histories[stage]
            for ax, key in zip(axes, ['loss', 'margin']):
                ax.plot([row['step'] for row in rows], [row[key] for row in rows], color=color, label=label)
        for ax, key in zip(axes, ['loss', 'margin']):
            ax.set(xlabel='Optimizer step', ylabel='DPO ' + key)
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=8)
        axes[1].axhline(0, color='#666666', linestyle='--', linewidth=.7)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=220)
        plt.close(fig)
    print('Validated 90 outputs and 220 optimizer steps; summary tables and curves saved.')
    for row in summaries:
        if row['type'] == 'total':
            print(row)


if __name__ == '__main__':
    main()
