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

STAGES = ['baseline','baseline_nf4','sft','dpo_original_b01','dpo_expanded_b01',
          'dpo_expanded_b005','dpo_expanded_b05','dpo60_lr1e5','sft120',
          'dpo120_lr1e5','sft120_eos','dpo120_eos_lr1e5']
BETAS = {'dpo_original_b01': .1, 'dpo_expanded_b01': .1,
         'dpo_expanded_b005': .05, 'dpo_expanded_b05': .5,
         'dpo60_lr1e5':.1,'dpo120_lr1e5':.1,'dpo120_eos_lr1e5':.1}


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
    parser.add_argument('--input-dir', type=Path, default=Path('results_corrected'))
    parser.add_argument('--output-dir', type=Path, default=Path('results_corrected'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    if (args.input_dir / 'baseline' / 'predictions.csv').exists():
        predictions = []
        for stage in STAGES:
            folder = stage if stage in ['baseline','baseline_nf4'] else stage+'_eval'
            predictions.extend(read_csv(args.input_dir / folder / 'predictions.csv'))
    else:
        predictions = read_csv(args.input_dir / 'all_predictions.csv')
    checked = []
    for row in predictions:
        guide, fact = metrics(row, row['prediction'])
        checked.append(dict(stage=row['stage'], id=row['id'], type=row['type'],
            img=row['img'], q=row['q'], gold=row['gold'], prediction=row['prediction'],
            guide_correct=guide, fact_correct=fact,
            generated_tokens=int(row['generated_tokens']),ended_by_eos=int(row['ended_by_eos']),
            reached_token_limit=int(row['reached_token_limit'])))
    summaries = []
    for stage in STAGES:
        rows = [row for row in checked if row['stage'] == stage]
        if len(rows) != 15:
            raise ValueError(f'{stage}: expected 15 outputs, got {len(rows)}')
        for kind in ['count', 'attr', 'exist', 'total']:
            group = rows if kind == 'total' else [row for row in rows if row['type'] == kind]
            summaries.append(dict(stage=stage, type=kind, n=len(group),
                guide_correct=sum(row['guide_correct'] for row in group),
                fact_correct=sum(row['fact_correct'] for row in group),
                ended_by_eos=sum(row['ended_by_eos'] for row in group),
                reached_token_limit=sum(row['reached_token_limit'] for row in group)))
    write_csv(out / 'results.csv', summaries)
    write_csv(out / 'all_predictions.csv', checked)
    histories = {}
    stats = {}
    for stage in STAGES[2:]:
        live = args.input_dir / stage / 'history.csv'
        rows = read_csv(live if live.exists() else args.input_dir / ('history_' + stage + '.csv'))
        for row in rows:
            for key in row:
                row[key] = int(row[key]) if key in ['step', 'epoch', 'sample_index'] else float(row[key])
        expected = 120 if stage in ['sft120','sft120_eos'] else 60 if stage == 'sft' else 40
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
    fig,ax=plt.subplots(figsize=(7.2,2.8))
    for stage,label,color in [('sft120','Original answer encoding','#31688e'),('sft120_eos','EOS supervision only','#35b779')]:
        rows=histories[stage];ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],label=label,color=color)
    ax.set(xlabel='Optimizer step',ylabel='SFT loss');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(figures/'extension_sft.png',dpi=220);plt.close(fig)
    if (args.input_dir/'heldout_predictions.csv').exists():
        independent=read_csv(args.input_dir/'heldout_predictions.csv');result=[]
        for row in independent:
            row['guide_correct'],row['fact_correct']=metrics(row,row['prediction'])
        for stage in dict.fromkeys(r['stage'] for r in independent):
            rows=[r for r in independent if r['stage']==stage];assert len(rows)==30
            for kind in ['count','attr','exist','total']:
                group=rows if kind=='total' else [r for r in rows if r['type']==kind]
                result.append(dict(stage=stage,type=kind,n=len(group),
                    guide_correct=sum(r['guide_correct'] for r in group),fact_correct=sum(r['fact_correct'] for r in group)))
        write_csv(out/'heldout_predictions.csv',independent);write_csv(out/'heldout_results.csv',result)
    print('Validated 180 original outputs and 580 optimizer steps; summary tables and curves saved.')
    for row in summaries:
        if row['type'] == 'total':
            print(row)


if __name__ == '__main__':
    main()
