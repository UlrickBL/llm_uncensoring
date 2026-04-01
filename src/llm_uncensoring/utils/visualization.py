"""
Visualisation utilities for all three experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


def _require_plot():
    if not _PLOT_AVAILABLE:
        raise ImportError("Install matplotlib and seaborn: pip install matplotlib seaborn")


# ---------------------------------------------------------------------------
# Patchscopes plots
# ---------------------------------------------------------------------------

def plot_refusal_scores_by_layer(
    stats: Dict,
    title: str = "Refusal score by layer (Patchscopes probe)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "plt.Figure":
    """
    Line plot of mean (± 1 std) refusal score at each layer.
    Highlights the layer of fastest increase (suspected guardrail onset).
    """
    _require_plot()
    layers = stats["layer_indices"]
    mean   = stats["mean_refusal"]
    std    = stats["std_refusal"]
    entropy= stats["mean_entropy"]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_refusal = "#e74c3c"
    color_entropy = "#3498db"

    ax1.plot(layers, mean, color=color_refusal, linewidth=2, label="Refusal score")
    ax1.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color_refusal)
    ax1.set_xlabel("Layer index", fontsize=12)
    ax1.set_ylabel("Mean refusal score", color=color_refusal, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color_refusal)

    ax2 = ax1.twinx()
    ax2.plot(layers, entropy, color=color_entropy, linewidth=1.5,
             linestyle="--", alpha=0.7, label="Entropy")
    ax2.set_ylabel("Mean entropy (nats)", color=color_entropy, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_entropy)

    # Mark guardrail onset (steepest increase)
    diffs = np.diff(mean)
    onset_idx = int(np.argmax(diffs)) + 1
    ax1.axvline(x=layers[onset_idx], color="gray", linestyle=":", linewidth=1.5,
                label=f"Max Δ at layer {layers[onset_idx]}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_refusal_heatmap(
    stats: Dict,
    prompt_labels: Optional[List[str]] = None,
    title: str = "Per-prompt refusal score heatmap",
    save_path: Optional[str] = None,
    show: bool = True,
    max_prompts: int = 30,
) -> "plt.Figure":
    """
    Heatmap of refusal_score[prompt, layer].
    """
    _require_plot()
    matrix = stats["refusal_matrix"][:max_prompts]   # [n_prompts, n_layers]
    layers = stats["layer_indices"]

    if prompt_labels is None:
        prompt_labels = [f"P{i}" for i in range(len(matrix))]
    prompt_labels = prompt_labels[:max_prompts]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) // 3), max(4, len(matrix) // 2)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=layers,
        yticklabels=prompt_labels,
        cmap="Reds",
        vmin=0,
        vmax=matrix.max(),
        linewidths=0.0,
        cbar_kws={"label": "Refusal score"},
    )
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Prompt", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Reduce x-tick density
    n_ticks = min(20, len(layers))
    step = max(1, len(layers) // n_ticks)
    ax.set_xticks(range(0, len(layers), step))
    ax.set_xticklabels(layers[::step], rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Intervention plot
# ---------------------------------------------------------------------------

def plot_intervention_comparison(
    baseline_refusal_rates: List[float],
    intervened_refusal_rates: List[float],
    modes: Optional[List[str]] = None,
    title: str = "Logit intervention: refusal rate comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> "plt.Figure":
    _require_plot()
    x = np.arange(len(baseline_refusal_rates))
    modes = modes or [f"run {i}" for i in x]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    ax.bar(x - width / 2, baseline_refusal_rates,  width, label="Baseline",    color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, intervened_refusal_rates, width, label="Intervened", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("Refusal rate", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    metrics: Dict,
    title: str = "RL self-distillation training",
    save_path: Optional[str] = None,
    show: bool = True,
    smooth_window: int = 20,
) -> "plt.Figure":
    _require_plot()
    from llm_uncensoring.utils.metrics import smooth

    steps  = metrics["step"]
    losses = metrics["loss"]
    rates  = metrics["answered_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, losses, alpha=0.3, color="#e74c3c", linewidth=0.8)
    if len(losses) >= smooth_window:
        s = smooth(losses, smooth_window)
        ax1.plot(steps[smooth_window - 1:], s, color="#e74c3c", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss")

    ax2.plot(steps, rates, alpha=0.3, color="#2ecc71", linewidth=0.8)
    if len(rates) >= smooth_window:
        s = smooth(rates, smooth_window)
        ax2.plot(steps[smooth_window - 1:], s, color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Answered rate")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_title("Fraction of prompts answered")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
