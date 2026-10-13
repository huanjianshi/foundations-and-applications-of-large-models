# Assignment 05 — SmolVLM QLoRA and DPO

A synthetic visual-question-answering experiment using **HuggingFaceTB/SmolVLM-256M-Instruct**. A hand-written training loop performs 4-bit NF4 QLoRA supervised fine-tuning (SFT), followed by direct preference optimization (DPO). The code includes 15 generated images, 15 questions, 12 original preference pairs and 5 additional pairs.

## Environment and model preparation

The checked run used Python 3.12, PyTorch 2.8.0+cu128, transformers 4.57.6, peft 0.18.0 and bitsandbytes 0.48.2. Full training requires a compatible NVIDIA GPU; this program does not silently replace NF4 training with CPU LoRA. A Blackwell GPU requires a compatible CUDA wheel, such as the CUDA 12.8 build below.

```bash
python -m venv .venv
# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1
# Or on Linux
# source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Download the model once. ONNX assets are not needed:

```bash
hf download HuggingFaceTB/SmolVLM-256M-Instruct --local-dir models/SmolVLM-256M-Instruct --include "*.json" "*.safetensors" "*.txt" "*.model" "*.jinja" --exclude "onnx/*"
```

The experiment loads the model **offline** from this local directory. If using another location, supply `--model-path`. For slow downloads, set `HF_HUB_DISABLE_XET=1`; an alternate endpoint such as the course-provided `https://hf-mirror.com` can be set during downloading. Model weights are not included in this repository.

## Usage

```bash
python hw5_dpo.py --mode make_data
python hw5_dpo.py --mode verify
python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1
python hw5_dpo.py --mode infer
python hw5_dpo.py --mode sft
python hw5_dpo.py --mode dpo --pairs expanded --beta 0.1
python hw5_dpo.py --mode eval
```

`sft` saves to `results/sft/adapter`; individual `dpo` saves to `results/dpo/adapter`. Default `eval` compares the baseline, SFT and the individually saved DPO adapter, or the expanded beta=0.1 adapter from the complete suite when an individual adapter is absent.

Run the complete experiment matrix and aggregate its outputs:

```bash
python hw5_dpo.py --mode all
python summarize_results.py --input-dir results --output-dir results
```

The suite generates the baseline, SFT, four independent DPO runs and evaluations of the five added questions. Each DPO branch reloads the **same SFT checkpoint**, rather than continuing from the previous DPO branch. Re-running the suite overwrites its run directories.

Redraw figures and check the uploaded results without downloading the model or training:

```bash
python summarize_results.py --input-dir artifacts --output-dir reproduced
```

## Controlled settings

| Setting | Value |
| --- | --- |
| Synthetic images / training seed | 42 / 42 |
| Image and processor size | 256×256 source images; one 512-pixel vision patch; no image splitting |
| Generation | Greedy, at most 24 new tokens; fp32 baseline for all adapter evaluations |
| LoRA | r=8, alpha=16; q/k/v/o language projections and matching q/k/v vision projections |
| SFT | 15 samples, 60 optimizer steps, batch=1, AdamW learning rate 1e-4, dropout=0.05 |
| DPO | 40 optimizer steps, batch=1, AdamW learning rate 5e-5, dropout disabled |
| DPO reference | Frozen NF4 base with its adapter disabled, not the SFT policy |
| Preference data comparison | 12 pairs versus 17 pairs, beta=0.1 |
| Beta scan | 17 pairs, beta=0.05 / 0.1 / 0.5 |

Only answer tokens contribute to SFT loss and sequence log probabilities. DPO uses **summed** answer-token log probabilities and `-logsigmoid(margin)`. Reference probabilities are computed without gradients and cached. The optimizer is ordinary AdamW, not a paged optimizer. `--mode verify` checks arithmetic, chosen/rejected gradient direction and the original substring scorer's existence false positive.

## Checked results

Two scoring conventions are retained: the course-guide Chinese substring matcher and a fact check that accepts equivalent English answers while rejecting contradictions and responses about a different object.

| Stage | Guide matcher | Fact check |
| --- | --- | --- |
| Baseline | 0/15 | 7/15 |
| SFT, 60 steps | 3/15 | 3/15 |
| DPO, 12 pairs, beta=0.1 | 4/15 | 4/15 |
| DPO, 17 pairs, beta=0.1 | 5/15 | 6/15 |
| DPO, 17 pairs, beta=0.05 | 4/15 | 5/15 |
| DPO, 17 pairs, beta=0.5 | 4/15 | 4/15 |

The baseline predominantly answers in English, so Chinese-only matching understates its task accuracy. SFT reduces training loss but worsens fact accuracy. Expanded DPO partially recovers that accuracy without exceeding the baseline. The beta=0.5 run has the lowest final DPO loss but lower accuracy than beta=0.1; losses across different beta values are scaled differently and cannot alone select a better policy.

The five added questions obtain 1/5 fact-correct answers with SFT and 2/5 after expanded DPO. The model still answers “有。” when asked about an absent blue circle. The three-question fidelity sample was reviewed and confirmed by the human user; it is too small to represent overall preference performance.

Training and evaluation questions overlap. These are classroom behavior comparisons, **not independent-test generalization measurements**. The final executable adds stricter scoring and a three-stage default evaluation after the recorded training; training and generation settings are unchanged. A 45-output evaluation rerun reproduced the recorded three-stage outputs exactly.

## Reproducibility artifacts

- [Aggregate results](artifacts/results.csv)
- [All 90 original-question outputs](artifacts/all_predictions.csv)
- [20 added-question outputs](artifacts/probe_predictions.csv)
- [Confirmed fidelity scores](artifacts/fidelity_scores.csv)
- [DPO arithmetic and gradient checks](artifacts/dpo_checks.json)
- [Parameter statistics](artifacts/parameter_stats.json)
- [Configuration and software versions](artifacts/experiment_config.json)
- [SFT history](artifacts/history_sft.csv)
- [Original-pair DPO history](artifacts/history_dpo_original_b01.csv)
- [Expanded-pair DPO history, beta=0.1](artifacts/history_dpo_expanded_b01.csv)
- [Expanded-pair DPO history, beta=0.05](artifacts/history_dpo_expanded_b005.csv)
- [Expanded-pair DPO history, beta=0.5](artifacts/history_dpo_expanded_b05.csv)
- [Figures](figures/)

The report, guide, template, lecture slides, model and adapter weights, local-path logs, caches and document-generation utilities are excluded.

## References

- [SmolVLM model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [QLoRA](https://arxiv.org/abs/2305.14314)
