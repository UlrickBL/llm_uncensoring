"""
Dataset loading and preprocessing for onepaneai/gpt-harmful-prompts-after-guardrails-evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset, Dataset


DATASET_ID = "onepaneai/gpt-harmful-prompts-after-guardrails-evaluation"


@dataclass
class HarmfulPromptsDataset:
    prompts: List[str]
    metadata: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def sample(self, n: int, seed: int = 42) -> "HarmfulPromptsDataset":
        rng = random.Random(seed)
        indices = rng.sample(range(len(self.prompts)), min(n, len(self.prompts)))
        return HarmfulPromptsDataset(
            prompts=[self.prompts[i] for i in indices],
            metadata=[self.metadata[i] for i in indices] if self.metadata else [],
        )

    def to_hf_dataset(self) -> Dataset:
        return Dataset.from_dict({"prompt": self.prompts, "metadata": self.metadata})


def load_harmful_prompts(
    split: str = "train",
    text_column: str = "prompt",
    max_samples: Optional[int] = None,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
) -> HarmfulPromptsDataset:
    """
    Load the harmful-prompts dataset and return a HarmfulPromptsDataset.

    The dataset may use different column names depending on the version.
    We probe common names and fall back gracefully.
    """
    ds = load_dataset(dataset_id, split=split)

    # Discover the prompt column
    column = _resolve_column(ds, preferred=text_column)

    prompts: List[str] = ds[column]
    # Collect remaining columns as metadata dicts
    meta_cols = [c for c in ds.column_names if c != column]
    metadata = [
        {c: ds[c][i] for c in meta_cols} for i in range(len(prompts))
    ]

    result = HarmfulPromptsDataset(prompts=prompts, metadata=metadata)

    if max_samples is not None and max_samples < len(result):
        result = result.sample(max_samples, seed=seed)

    return result


def _resolve_column(ds: Dataset, preferred: str) -> str:
    if preferred in ds.column_names:
        return preferred
    # Common fallback names
    for candidate in ["prompt", "text", "question", "input", "instruction"]:
        if candidate in ds.column_names:
            return candidate
    raise ValueError(
        f"Cannot find a text column in dataset. Available: {ds.column_names}. "
        f"Set `text_column` in config to the correct name."
    )


def build_chat_prompt(
    prompt: str,
    tokenizer,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Apply the model's chat template to a raw prompt string."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
