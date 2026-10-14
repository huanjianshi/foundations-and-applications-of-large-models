# Assignment 05 — SmolVLM QLoRA and DPO

This corrected experiment uses HuggingFaceTB/SmolVLM-256M-Instruct for synthetic visual question answering. Real runs include the required SFT60/DPO40 matrix, 12 versus 17 preference pairs, beta=0.05/0.1/0.5, and explicitly separated training-budget, learning-rate and EOS extensions.

## Correction

The first publication loaded original FP32 base weights for NF4-trained adapters and incorrectly interpreted the resulting scores as SFT deterioration. Matched NF4 inference gives SFT60 **11/15** and expanded DPO40 **10/15**, rather than 3/15 and 6/15. Those previous conclusions are withdrawn. FP32 adapter evaluations are retained only as explicit precision controls.

The SFT120 EOS-only extension and its DPO40 continuation at lr=1e-5 achieve **15/15** on the training-overlap questions, with **15/15** generations ending in EOS. This does not establish independent-test generalization. All ten training runs pass save/reload comparisons for 15 raw generations and answer losses.

## Preparation

The checked environment is Python 3.12, torch 2.8.0+cu128, transformers 4.57.6, peft 0.18.0 and bitsandbytes 0.48.2. A compatible NVIDIA CUDA GPU is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
hf download HuggingFaceTB/SmolVLM-256M-Instruct --local-dir models/SmolVLM-256M-Instruct --include "*.json" "*.safetensors" "*.txt" "*.model" "*.jinja" --exclude "onnx/*"
```

Model loading is offline; use `--model-path` for another local cache. Model and adapter weights are excluded.

## Reproduction

```bash
python hw5_dpo.py --mode verify
python hw5_dpo.py --mode all
python run_corrected_experiments.py --resume
python summarize_results.py --input-dir results_corrected --output-dir summarized
```

For the complete matrix from a fresh directory, run `python run_corrected_experiments.py` without `--resume`. Resume keeps completed training checkpoints and reruns evaluations; a fresh full run replaces its named result directories. Each DPO branch starts from its declared SFT checkpoint. Default `eval` uses corrected SFT60 and expanded DPO40. EOS120 remains a separately labeled extension.

Individual extension commands:

```bash
python hw5_dpo.py --mode sft --steps 120 --supervise-eos --roundtrip-check --output results_corrected/sft120_eos
python hw5_dpo.py --mode dpo --steps 40 --learning-rate 1e-5 --supervise-eos --checkpoint-every 10 --roundtrip-check --sft-adapter results_corrected/sft120_eos/adapter --output results_corrected/dpo120_eos_lr1e5
python hw5_dpo.py --mode eval --adapter results_corrected/dpo120_eos_lr1e5/adapter --output results_corrected/final_eval
```

To validate uploaded measurements without downloading the model or training:

```bash
python summarize_results.py --input-dir artifacts --output-dir reproduced
```

## Controls and metrics

All runs use seed42, original 256x256 images, a single 512-pixel vision patch without splitting, greedy generation and at most 24 new tokens. LoRA uses r=8, alpha=16, q/k/v/o projections (including matching vision q/k/v), SFT dropout0.05 and batch1. Required SFT uses 60 updates at lr=1e-4; required DPO uses 40 at lr=5e-5 with dropout disabled. The reference is the frozen NF4 base with its adapter disabled, following the guide. It is not silently replaced with a frozen SFT policy. Summed answer-token log probabilities and stable `-logsigmoid(margin)` are used.

`--eval-precision auto` uses FP32 for an unadapted baseline and NF4 for adapters. Explicit `fp32` adapter inference records the mismatch. `--supervise-eos` only appends the correct EOS to the original answer encoding; it does not change the assistant prefix. No generated answers are edited or replaced to improve scores.

Guide substring matching is preserved alongside a target-aware task check. The task check rejects unrelated text/table answers, yes/no contradictions and mismatched colors, and accepts equivalent English shape/number answers. Fidelity remains separate from answer correctness. The nine corrected answers were reviewed and the suggested scores confirmed by the user.

## Original 15 questions

| Stage | Guide match | Task correct | Ended in EOS |
|---|---:|---:|---:|
| baseline | 0/15 | 7/15 | 9/15 |
| baseline_nf4 | 0/15 | 7/15 | 14/15 |
| sft | 11/15 | 11/15 | 14/15 |
| dpo_original_b01 | 9/15 | 10/15 | 15/15 |
| dpo_expanded_b01 | 10/15 | 10/15 | 14/15 |
| dpo_expanded_b005 | 10/15 | 10/15 | 14/15 |
| dpo_expanded_b05 | 10/15 | 10/15 | 15/15 |
| dpo60_lr1e5 | 11/15 | 11/15 | 15/15 |
| sft120 | 15/15 | 15/15 | 13/15 |
| dpo120_lr1e5 | 15/15 | 15/15 | 13/15 |
| sft120_eos | 15/15 | 15/15 | 15/15 |
| dpo120_eos_lr1e5 | 15/15 | 15/15 | 15/15 |

Beta branches all score 10/15 under matched NF4 inference; the former claim that beta0.1 beats beta0.5 on accuracy is withdrawn. Losses across beta values are scaled differently. Required DPO can hurt chosen answer probabilities while suppressing rejected probabilities more strongly; lower loss alone does not establish better generation.

## Independent test

Thirty new images and questions are generated with seed20261014: 10 count, 10 attribute, 10 existence (balanced yes/no). They use nonoverlapping jittered layouts and new wording. The final model/configuration was fixed before test evaluation; test questions never enter training or checkpoint selection. Both visual layout and wording change, so this is a limited distribution-shift test.

| Stage | Task correct |
|---|---:|
| baseline | 9/30 |
| baseline_nf4 | 7/30 |
| sft | 10/30 |
| dpo_expanded_b01 | 10/30 |
| sft120_eos | 7/30 |
| dpo120_eos_lr1e5 | 7/30 |

The original 15 questions overlap SFT training. Their 15/15 score must not be presented as generalization accuracy. All test failures are retained.

## Evidence

- [Aggregate original-question results](artifacts/results.csv) and [180 raw outputs](artifacts/all_predictions.csv)
- [Independent-test results](artifacts/heldout_results.csv), [180 raw test outputs](artifacts/heldout_predictions.csv), and [fixed test manifest](artifacts/heldout_manifest.json)
- [25 added-question outputs](artifacts/probe_predictions.csv)
- [DPO checkpoints](artifacts/checkpoint_results.csv) and [FP32 precision-control outputs](artifacts/precision_control_predictions.csv)
- [Fidelity scores and confirmation status](artifacts/fidelity_scores.csv)
- [Roundtrip checks](artifacts/roundtrip_checks.json), [DPO arithmetic checks](artifacts/dpo_checks.json), [parameters](artifacts/parameter_stats.json), [configuration](artifacts/experiment_config.json)
- [Histories](artifacts/) contain 580 optimizer updates; [figures](figures/) show recorded trajectories.

The report, guide, template, lecture slides, student identifiers, local-path logs and caches are excluded. Training manifests record actual source hashes; required experiments precede the final target-aware scorer refinement, which does not change model updates or generation.

## References

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PEFT quantization documentation](https://huggingface.co/docs/peft/developer_guides/quantization)
