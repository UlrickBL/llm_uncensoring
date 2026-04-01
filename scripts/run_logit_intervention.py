"""
Experiment 2 — Logit intervention.

Generates completions with and without refusal token suppression and compares
the refusal rates and response quality.

Usage:
    python scripts/run_logit_intervention.py
    python scripts/run_logit_intervention.py --mode adaptive --n_prompts 50
    python scripts/run_logit_intervention.py --contrastive   # build refusal set contrastively
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from llm_uncensoring.data.dataset import load_harmful_prompts, build_chat_prompt
from llm_uncensoring.models.loader import load_model_and_tokenizer
from llm_uncensoring.experiments.logit_intervention import (
    build_refusal_token_ids,
    build_contrastive_refusal_ids,
    generate_baseline_and_intervened,
)
from llm_uncensoring.utils.metrics import intervention_stats
from llm_uncensoring.utils.visualization import plot_intervention_comparison


def parse_args():
    p = argparse.ArgumentParser(description="Logit intervention experiment")
    p.add_argument("--config",       default="configs/logit_intervention.yaml")
    p.add_argument("--mode",         default=None,
                   choices=["suppress", "penalty", "adaptive"])
    p.add_argument("--n_prompts",    type=int, default=None)
    p.add_argument("--contrastive",  action="store_true",
                   help="Build refusal token set contrastively")
    p.add_argument("--output_dir",   default=None)
    p.add_argument("--no_plot",      action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
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

    if args.mode:
        cfg["intervention"]["mode"] = args.mode
    if args.n_prompts:
        cfg["evaluation"]["n_prompts"] = args.n_prompts
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

    print(f"[2/4] Loading dataset")
    dataset = load_harmful_prompts(
        split=cfg["dataset"]["split"],
        text_column=cfg["dataset"]["text_column"],
        max_samples=cfg["evaluation"]["n_prompts"],
    )
    print(f"      Loaded {len(dataset)} prompts")

    print(f"[3/4] Building refusal token set")
    strategy = "contrastive" if args.contrastive else cfg["refusal_tokens"]["strategy"]

    if strategy == "contrastive":
        refusal_ids = build_contrastive_refusal_ids(
            model=model,
            tokenizer=tokenizer,
            prompts=dataset.prompts,
            n_samples=cfg["refusal_tokens"]["contrastive_n_samples"],
            top_k_tokens=cfg["refusal_tokens"]["contrastive_top_k"],
        )
    else:
        refusal_ids = build_refusal_token_ids(
            tokenizer=tokenizer,
            phrases=cfg["refusal_tokens"]["phrases"],
        )

    refusal_strs = [tokenizer.decode([tid]) for tid in refusal_ids]
    print(f"      {len(refusal_ids)} refusal tokens: {refusal_strs[:10]}{'...' if len(refusal_strs) > 10 else ''}")

    prompts = [build_chat_prompt(p, tokenizer) for p in dataset.prompts]

    print(f"[4/4] Generating (baseline + intervened) with mode={cfg['intervention']['mode']!r}")
    results = generate_baseline_and_intervened(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        refusal_phrases=cfg["refusal_tokens"]["phrases"],
        mode=cfg["intervention"]["mode"],
        penalty_value=cfg["intervention"]["penalty_value"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        generate_baseline=cfg["evaluation"]["generate_baseline"],
    )

    baselines   = [r.baseline  or "" for r in results]
    intervened  = [r.intervened          for r in results]

    stats = intervention_stats(baselines, intervened)

    print("\n=== Logit Intervention Summary ===")
    print(f"  Baseline refusal rate   : {stats['baseline_refusal_rate']:.1%}")
    print(f"  Intervened refusal rate : {stats['intervened_refusal_rate']:.1%}")
    print(f"  Absolute reduction      : {stats['absolute_reduction']:.1%}")
    print(f"  Relative reduction      : {stats['relative_reduction']:.1%}")
    print(f"  Avg words (baseline)    : {stats['avg_words_baseline']:.0f}")
    print(f"  Avg words (intervened)  : {stats['avg_words_intervened']:.0f}")

    # Save outputs
    out_records = [
        {
            "prompt": r.prompt,
            "baseline": r.baseline,
            "intervened": r.intervened,
        }
        for r in results
    ]
    out_file = output_dir / f"intervention_{cfg['intervention']['mode']}_results.json"
    with open(out_file, "w") as f:
        json.dump({"stats": stats, "results": out_records}, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    if not args.no_plot:
        plot_intervention_comparison(
            baseline_refusal_rates=[stats["baseline_refusal_rate"]],
            intervened_refusal_rates=[stats["intervened_refusal_rate"]],
            modes=[cfg["intervention"]["mode"]],
            save_path=str(output_dir / "intervention_comparison.png"),
            show=False,
        )

    # Print a few examples
    print("\n=== Sample Outputs (first 3) ===")
    for i, r in enumerate(results[:3]):
        print(f"\n--- Prompt {i+1} ---")
        print(f"  PROMPT     : {r.prompt[:120]}...")
        if r.baseline:
            print(f"  BASELINE   : {r.baseline[:200]}")
        print(f"  INTERVENED : {r.intervened[:200]}")


if __name__ == "__main__":
    main()
