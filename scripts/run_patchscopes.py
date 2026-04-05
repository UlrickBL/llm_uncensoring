"""
Experiment 1 — Patchscopes layer probing.

For each prompt, extracts the intermediate hidden state at every transformer
layer, applies (final_norm + LM head) as the "scope", and records the
probability of refusal-related tokens.  Produces a per-layer refusal score
curve and a heatmap across prompts.

Usage:
    python scripts/run_patchscopes.py
    python scripts/run_patchscopes.py --config configs/patchscopes.yaml
    python scripts/run_patchscopes.py --n_prompts 5 --model Qwen/Qwen3-4B
    python scripts/run_patchscopes.py --model-config configs/models/ministral.yaml
    python scripts/run_patchscopes.py --scorer cosine
    python scripts/run_patchscopes.py --scorer reranker
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from llm_uncensoring.data.dataset import load_harmful_prompts, build_chat_prompt
from llm_uncensoring.models.loader import load_model_and_tokenizer
from llm_uncensoring.experiments.patchscopes import run_patchscopes, build_scorer_from_config
from llm_uncensoring.utils.metrics import compute_patchscopes_stats
from llm_uncensoring.utils.visualization import plot_refusal_scores_by_layer, plot_refusal_heatmap


def parse_args():
    p = argparse.ArgumentParser(description="Patchscopes layer probe experiment")
    p.add_argument("--config",        default="configs/patchscopes.yaml")
    p.add_argument("--model-config",  default=None,
                   help="Path to a model preset yaml (e.g. configs/models/ministral.yaml)")
    p.add_argument("--model",         default=None, help="Override model name directly")
    p.add_argument("--scorer",        default=None,
                   choices=["vocab_list", "cosine", "reranker"],
                   help="Override refusal scorer strategy")
    p.add_argument("--n_prompts",     type=int, default=None)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--no_plot",       action="store_true")
    return p.parse_args()


def load_config(path: str, model_config_path: str = None) -> dict:
    base_path = Path(path).parent / "base.yaml"
    cfg: dict = {}
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}
    with open(path) as f:
        cfg.update(yaml.safe_load(f) or {})
    # Overlay model-specific config (e.g. configs/models/ministral.yaml)
    if model_config_path:
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f) or {}
        for key, val in model_cfg.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, model_config_path=args.model_config)

    # CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
    if args.scorer:
        cfg["refusal"]["strategy"] = args.scorer
    if args.n_prompts:
        cfg["analysis"]["n_prompts"] = args.n_prompts
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading model: {cfg['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
        attn_implementation=cfg["model"].get("attn_implementation"),
    )

    print(f"[2/4] Loading dataset: {cfg['dataset']['name']}")
    dataset = load_harmful_prompts(
        split=cfg["dataset"]["split"],
        text_column=cfg["dataset"]["text_column"],
        max_samples=cfg["analysis"]["n_prompts"],
    )
    print(f"      Loaded {len(dataset)} prompts")

    prompts = [build_chat_prompt(p, tokenizer) for p in dataset.prompts]
    raw_prompts = dataset.prompts

    print(f"[3/4] Running Patchscopes probe on {len(prompts)} prompts...")
    print(f"      Refusal scorer strategy: {cfg['refusal']['strategy']!r}")
    scorer = build_scorer_from_config(cfg["refusal"], tokenizer=tokenizer)

    layer_indices = cfg["probe"].get("layer_indices", None)
    results = run_patchscopes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        questions=raw_prompts,
        scorer=scorer,
        layer_indices=layer_indices,
        top_k=cfg["probe"]["top_k"],
        apply_final_norm=cfg["probe"]["apply_final_norm"],
    )

    print(f"[4/4] Computing stats and saving results...")
    stats = compute_patchscopes_stats(results)

    # Save raw results as JSON
    raw_out = []
    for res, raw in zip(results, raw_prompts):
        raw_out.append({
            "prompt": raw,
            "guardrail_onset_layer": res.guardrail_onset_layer(),
            "layers": [
                {
                    "layer_idx": lr.layer_idx,
                    "refusal_score": lr.refusal_score,
                    "entropy": lr.entropy,
                    "top_tokens": lr.top_tokens[:5],
                    "top_probs": lr.top_probs[:5],
                }
                for lr in res.layer_results
            ],
        })
    with open(output_dir / "patchscopes_results.json", "w") as f:
        json.dump(raw_out, f, indent=2)
    print(f"      Results saved to {output_dir / 'patchscopes_results.json'}")

    # Print summary
    import numpy as np
    onset_layers = stats["onset_layers"]
    print("\n=== Patchscopes Summary ===")
    print(f"  Median guardrail onset layer : {np.median(onset_layers):.0f}")
    print(f"  Mean   guardrail onset layer : {np.mean(onset_layers):.1f} ± {np.std(onset_layers):.1f}")
    peak_layer = stats["layer_indices"][np.argmax(stats["mean_refusal"])]
    print(f"  Peak refusal score at layer  : {peak_layer}")

    if not args.no_plot:
        fmt = cfg["output"].get("plot_format", "png")
        plot_refusal_scores_by_layer(
            stats,
            save_path=str(output_dir / f"refusal_by_layer.{fmt}"),
            show=False,
        )
        short_labels = [p[:40] + "..." if len(p) > 40 else p for p in raw_prompts]
        plot_refusal_heatmap(
            stats,
            prompt_labels=short_labels,
            save_path=str(output_dir / f"refusal_heatmap.{fmt}"),
            show=False,
        )
        print(f"      Plots saved to {output_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
