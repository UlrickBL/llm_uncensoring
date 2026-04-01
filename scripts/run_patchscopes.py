"""
Experiment 1 — Patchscopes layer probing.

For each prompt, extracts the intermediate hidden state at every transformer
layer, applies (final_norm + LM head) as the "scope", and records the
probability of refusal-related tokens.  Produces a per-layer refusal score
curve and a heatmap across prompts.

Usage:
    python scripts/run_patchscopes.py
    python scripts/run_patchscopes.py --config configs/patchscopes.yaml
    python scripts/run_patchscopes.py --n_prompts 5 --model Qwen/Qwen2.5-2B-Instruct
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
from llm_uncensoring.experiments.patchscopes import run_patchscopes
from llm_uncensoring.utils.metrics import compute_patchscopes_stats
from llm_uncensoring.utils.visualization import plot_refusal_scores_by_layer, plot_refusal_heatmap


def parse_args():
    p = argparse.ArgumentParser(description="Patchscopes layer probe experiment")
    p.add_argument("--config",     default="configs/patchscopes.yaml")
    p.add_argument("--model",      default=None, help="Override model name")
    p.add_argument("--n_prompts",  type=int, default=None, help="Override number of prompts")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_plot",    action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Flatten defaults hierarchy if present
    base_path = Path(path).parent / "base.yaml"
    if base_path.exists():
        with open(base_path) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.model:
        cfg["model"]["name"] = args.model
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
    layer_indices = cfg["probe"].get("layer_indices", None)
    results = run_patchscopes(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        refusal_phrases=cfg["refusal"]["phrases"],
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
