# Assignment 03 — Minimal LLM on CPU

A character-level decoder-only Transformer implemented in PyTorch and trained from scratch on a small built-in Tang-poetry corpus. The program supports the required controlled experiments, checkpoint reuse, configurable sampling, and a no-repeat 3-gram extension.

## Environment

- Python 3.12+
- PyTorch 2.x (CPU is sufficient)
- Matplotlib 3.5+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run one configuration (the guide-compatible interface is retained):

```bash
python min_llm.py --iters 800
```

Run the complete experiment suite:

```bash
python min_llm.py --suite --output_dir runs
```

Important options include `--lr`, `--n_layer`, `--n_embd`, `--block_size`, `--no_pos`, `--temperature`, `--top_k`, `--no_repeat_ngram`, `--seed`, and `--threads`.

## Experiment matrix

| Group | Changed variable | Setting(s) |
| --- | --- | --- |
| Baseline | — | 2 layers, 128-dimensional embeddings, context 128, learning rate 1e-3, 2000 iterations |
| Exp. 1 | Learning rate | 1e-2, 1e-4 |
| Exp. 2 | Number of layers | 1, 4 |
| Exp. 3 | Embedding size | 64, 256 |
| Exp. 4 | Context length | 32 |
| Exp. 5 | Position embedding | Disabled |
| Exp. 6 | Sampling | Five temperature/top-k combinations using the saved baseline |
| Extension | Repetition control | No-repeat 3-gram decoding with matched seeds |

All training configurations use the same corpus and random seed unless the sampled output seed is the controlled variable.

## Reproduced headline results

The checked CPU run used PyTorch 2.14.0+cpu. The baseline contains **502,656 parameters**; its loss decreased from **6.5627** to **0.0263** in 2000 iterations. Full-precision measurements are provided in [`artifacts/results.csv`](artifacts/results.csv).

- [Training results](artifacts/results.csv)
- [Sampling results](artifacts/sampling_results.csv)
- [Variant samples](artifacts/variant_samples.csv)
- [Loss curves](figures/)

The model checkpoint, local logs, report document, assignment guide, and report template are intentionally excluded. They are either reproducible from the script or not appropriate for a public source repository.
