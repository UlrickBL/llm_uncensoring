# LLM Uncensoring — Research Codebase

> Research into where and how guardrails are encoded inside autoregressive LLMs, and how they can be bypassed or removed.
> Manuscript in preparation.

---

## Overview

Three complementary experiments attack the guardrail problem from different angles:

| # | Experiment | Question |
|---|-----------|----------|
| 1 | **Patchscopes layer probe** | *At which transformer layer does the model start predicting refusal tokens?* |
| 2 | **Logit intervention** | *If we force refusal tokens to zero probability, what lies underneath?* |
| 3 | **RL self-distillation** | *Can we train the model out of its refusal behaviour via online reward + self-reflection?* |

**Base model**: `Qwen/Qwen2.5-2B-Instruct`  
**Dataset**: [`onepaneai/gpt-harmful-prompts-after-guardrails-evaluation`](https://huggingface.co/datasets/onepaneai/gpt-harmful-prompts-after-guardrails-evaluation)

---

## Repository Structure

```
llm_uncensoring/
├── configs/
│   ├── base.yaml                  # Shared: model, dataset, generation params
│   ├── patchscopes.yaml           # Experiment 1 config
│   ├── logit_intervention.yaml    # Experiment 2 config
│   └── rl_distillation.yaml       # Experiment 3 config
│
├── src/llm_uncensoring/
│   ├── data/
│   │   └── dataset.py             # HF dataset loader + chat-template wrapper
│   ├── models/
│   │   └── loader.py              # Architecture-agnostic model/tokenizer loading
│   ├── experiments/
│   │   ├── patchscopes/
│   │   │   └── layer_probe.py     # LayerProbe, LayerProbeResult, run_patchscopes
│   │   ├── logit_intervention/
│   │   │   ├── refusal_tokens.py  # Build refusal token ID set (hardcoded or contrastive)
│   │   │   └── processor.py       # RefusalSuppressionProcessor, AdaptiveRefusalProcessor
│   │   └── rl_distillation/
│   │       ├── judge.py           # HeuristicJudge, LLMJudge
│   │       ├── reflector.py       # SelfReflector — generates post-reflection answers
│   │       └── trainer.py         # SelfDistillationTrainer (online DPO / SFT / KL)
│   └── utils/
│       ├── metrics.py             # refusal_rate, patchscopes_stats, intervention_stats
│       └── visualization.py       # matplotlib plots for all three experiments
│
└── scripts/
    ├── run_patchscopes.py
    ├── run_logit_intervention.py
    └── run_rl_distillation.py
```

---

## Setup

```bash
pip install -r requirements.txt
# Optionally install the package in editable mode:
pip install -e .
```

Flash Attention 2 (recommended for speed):
```bash
pip install flash-attn --no-build-isolation
```

---

## Experiment 1 — Patchscopes Layer Probe

### Idea

Inspired by [Patchscopes (Ghandeharioun et al., ICML 2024)](https://arxiv.org/abs/2401.06102).  
For each layer `l`, we intercept the hidden state `h_l` at the last token position and apply the model's own `(final_norm, lm_head)` pair as the *scope* — projecting the intermediate representation into vocabulary space.

We track the **refusal score** per layer: the total probability mass on tokens associated with refusal phrases (e.g. `"I"`, `"Sorry"`, `"cannot"`).  A spike in this score pinpoints the layer(s) where the guardrail information is encoded.

### Run

```bash
python scripts/run_patchscopes.py
# Specific overrides:
python scripts/run_patchscopes.py --n_prompts 10 --model Qwen/Qwen2.5-2B-Instruct
```

### Outputs (`outputs/patchscopes/`)

| File | Contents |
|------|----------|
| `patchscopes_results.json` | Per-prompt, per-layer: refusal score, entropy, top-5 tokens |
| `refusal_by_layer.png` | Mean ± std refusal score + entropy across layers |
| `refusal_heatmap.png` | Heatmap of refusal score [prompts × layers] |

### Key config knobs (`configs/patchscopes.yaml`)

```yaml
probe:
  apply_final_norm: true   # apply model's final LayerNorm before LM head
  top_k: 20                # decoded top-k tokens per layer

refusal:
  strategy: "vocab_list"   # or "computed" to derive from refused completions
  phrases: [...]           # phrases whose first subword token marks a refusal
```

---

## Experiment 2 — Logit Intervention

### Idea

If guardrail behaviour manifests as high probability on a small set of *refusal tokens*, we can test whether meaningful content is hiding just behind them by **zeroing those logits** at every decoding step.

Three modes:
- `suppress` — hard zero (logit → −∞) on every step
- `penalty`  — soft penalty (subtract a large scalar)
- `adaptive` — activates only after the first generated token is a refusal token (more surgical)

We also support a **contrastive strategy** to build the refusal token set: generate completions with a normal vs. a permissive system prompt, then extract tokens significantly over-represented in refused outputs.

### Run

```bash
python scripts/run_logit_intervention.py
# With contrastive token detection:
python scripts/run_logit_intervention.py --contrastive
# Adaptive mode:
python scripts/run_logit_intervention.py --mode adaptive
```

### Outputs (`outputs/logit_intervention/`)

| File | Contents |
|------|----------|
| `intervention_<mode>_results.json` | Per-prompt: baseline and intervened response, aggregate stats |
| `intervention_comparison.png` | Bar chart: refusal rate before vs after |

### Key config knobs (`configs/logit_intervention.yaml`)

```yaml
intervention:
  mode: "suppress"         # suppress | penalty | adaptive
  penalty_value: 100.0     # used in "penalty" mode

refusal_tokens:
  strategy: "hardcoded"    # or "contrastive"
  contrastive_top_k: 50    # tokens to keep when strategy=contrastive
```

---

## Experiment 3 — Online RL + Self-Distillation

### Idea

An online training loop that:

1. **Samples** a prompt and generates a response.
2. **Judges** the response: `ANSWERED` or `DODGED`?
3. **Self-reflects**: if dodged, the model's own output is extended with a
   reflection directive (`"I just refused — let me try again:"`) and re-sampled.
4. **Trains** on the (dodged, reflected) pair using one of three objectives:

| Mode | Objective | Signal |
|------|-----------|--------|
| `online_dpo` | Online DPO loss | chosen=reflected, rejected=dodged |
| `sft` | Cross-entropy | supervised on reflected answer |
| `kl` | KL divergence | match reflected token distribution |

LoRA is applied by default to keep memory usage tractable.

### Run

```bash
python scripts/run_rl_distillation.py
# DPO with LLM judge:
python scripts/run_rl_distillation.py --mode online_dpo --judge llm
# Quick SFT test (no LoRA, 50 steps):
python scripts/run_rl_distillation.py --mode sft --n_steps 50 --no_lora
```

### Outputs (`outputs/rl_distillation/`)

| File | Contents |
|------|----------|
| `checkpoint-<step>/` | LoRA adapter checkpoints |
| `training_metrics.json` | Per-step loss and answered rate |
| `training_curves.png` | Loss + answered rate plots |

### Key config knobs (`configs/rl_distillation.yaml`)

```yaml
judge:
  mode: "heuristic"        # heuristic | llm
  heuristic_threshold: 0.5 # fraction of matched patterns to count as dodged

training:
  mode: "online_dpo"       # online_dpo | sft | kl
  dpo_beta: 0.1            # KL regularisation in DPO
  lora:
    enabled: true
    r: 16
```

---

## Programmatic API

```python
from llm_uncensoring.models.loader import load_model_and_tokenizer
from llm_uncensoring.data.dataset import load_harmful_prompts, build_chat_prompt
from llm_uncensoring.experiments.patchscopes import run_patchscopes
from llm_uncensoring.utils.metrics import compute_patchscopes_stats
from llm_uncensoring.utils.visualization import plot_refusal_scores_by_layer

model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-2B-Instruct")
dataset = load_harmful_prompts(max_samples=20)
prompts = [build_chat_prompt(p, tokenizer) for p in dataset.prompts]

results = run_patchscopes(model, tokenizer, prompts)
stats   = compute_patchscopes_stats(results)
plot_refusal_scores_by_layer(stats)
```

```python
from llm_uncensoring.experiments.logit_intervention import generate_baseline_and_intervened

results = generate_baseline_and_intervened(model, tokenizer, prompts, mode="suppress")
for r in results:
    print("Baseline :", r.baseline[:100])
    print("Intervened:", r.intervened[:100])
```

```python
from llm_uncensoring.experiments.rl_distillation import SelfDistillationTrainer, TrainerConfig

config  = TrainerConfig(mode="online_dpo", n_steps=100)
trainer = SelfDistillationTrainer(model, tokenizer, dataset, config)
metrics = trainer.train()
```

---

## Notes

- **Model name**: The user requested `Qwen3.5 2B`. The closest publicly available
  model as of writing is `Qwen/Qwen2.5-2B-Instruct`. Update `configs/base.yaml`
  if a newer Qwen3 checkpoint is released.
- **Memory**: A 2B model in bfloat16 fits comfortably on a 10 GB GPU. LoRA reduces
  training peak memory further. For 7B+ models, enable `load_in_4bit: true`.
- **Ethics**: This codebase is intended for academic research into LLM safety
  and the mechanistic interpretation of guardrails. The techniques developed here
  are dual-use; responsible disclosure practices apply.

---

## Citation

If you use this codebase, please cite the associated paper (forthcoming) and:

```bibtex
@inproceedings{ghandeharioun2024patchscopes,
  title     = {Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models},
  author    = {Ghandeharioun, Asma and others},
  booktitle = {ICML},
  year      = {2024}
}
```
