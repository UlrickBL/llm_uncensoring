"""
Experiment 3 — Online RL with self-reflection and distillation.

Pipeline:
  1. Sample a prompt from the harmful-prompts dataset.
  2. Generate a response with the current model.
  3. Judge: ANSWERED or DODGED?
  4. If DODGED: run self-reflection to obtain a better response.
  5. Build training signal (DPO / SFT / KL) from the (dodged, reflected) pair.
  6. Gradient step via LoRA.

The model is nudged toward answering questions rather than deflecting them.

Usage:
    python scripts/run_rl_distillation.py
    python scripts/run_rl_distillation.py --mode sft --n_steps 200
    python scripts/run_rl_distillation.py --judge llm --mode online_dpo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml

from llm_uncensoring.data.dataset import load_harmful_prompts
from llm_uncensoring.models.loader import load_model_and_tokenizer
from llm_uncensoring.experiments.rl_distillation import SelfDistillationTrainer, TrainerConfig
from llm_uncensoring.experiments.rl_distillation.trainer import LoraSettings
from llm_uncensoring.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="RL self-distillation experiment")
    p.add_argument("--config",     default="configs/rl_distillation.yaml")
    p.add_argument("--mode",       default=None,
                   choices=["online_dpo", "sft", "kl"])
    p.add_argument("--judge",      default=None,
                   choices=["heuristic", "llm"])
    p.add_argument("--n_steps",    type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_lora",    action="store_true")
    p.add_argument("--no_plot",    action="store_true")
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
        cfg["training"]["mode"] = args.mode
    if args.judge:
        cfg["judge"]["mode"] = args.judge
    if args.n_steps:
        cfg["training"]["n_steps"] = args.n_steps
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading model: {cfg['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
        attn_implementation=cfg["model"].get("attn_implementation"),
    )

    print(f"[2/3] Loading dataset")
    dataset = load_harmful_prompts(
        split=cfg["dataset"]["split"],
        text_column=cfg["dataset"]["text_column"],
        max_samples=cfg["dataset"].get("max_samples"),
    )
    print(f"      Loaded {len(dataset)} prompts")

    lora_cfg = cfg["training"]["lora"]
    trainer_config = TrainerConfig(
        mode=cfg["training"]["mode"],
        n_steps=cfg["training"]["n_steps"],
        batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        save_every_n_steps=cfg["training"]["save_every_n_steps"],
        output_dir=cfg["output"]["dir"],
        dpo_beta=cfg["training"]["dpo_beta"],
        dpo_loss_type=cfg["training"]["dpo_loss_type"],
        kl_temperature=cfg["training"]["kl_temperature"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        lora=LoraSettings(
            enabled=lora_cfg["enabled"] and not args.no_lora,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
        ),
    )

    reflection_prompt = cfg["reflection"].get("system_prompt", "")

    print(f"[3/3] Training with mode={trainer_config.mode!r}, "
          f"judge={cfg['judge']['mode']!r}, steps={trainer_config.n_steps}")

    trainer = SelfDistillationTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=trainer_config,
        judge_mode=cfg["judge"]["mode"],
        reflection_prompt=reflection_prompt,
    )

    metrics = trainer.train()

    # Save metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTraining complete. Metrics saved to {metrics_file}")

    import numpy as np
    print(f"\n=== Training Summary ===")
    print(f"  Final loss          : {metrics['loss'][-1]:.4f}")
    print(f"  Final answered rate : {metrics['answered_rate'][-1]:.1%}")
    print(f"  Mean answered rate  : {np.mean(metrics['answered_rate']):.1%}")

    if not args.no_plot:
        plot_training_curves(
            metrics,
            save_path=str(output_dir / "training_curves.png"),
            show=False,
        )


if __name__ == "__main__":
    main()
