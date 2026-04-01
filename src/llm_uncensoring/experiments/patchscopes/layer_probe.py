"""
Patchscopes-style layer probing.

For each transformer layer l, we intercept the hidden state h_l at the last
token position, apply the model's own final LayerNorm + LM head (the "scope"),
and decode the resulting probability distribution.

This lets us track at which depth the model's internal representation starts
predicting refusal tokens — revealing where the guardrail logic is encoded.

Reference:
    Ghandeharioun et al., "Patchscopes: A Unifying Framework for Inspecting
    Hidden Representations of Language Models", ICML 2024.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_uncensoring.models.loader import (
    get_final_norm,
    get_lm_head,
    get_transformer_layers,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    layer_idx: int
    top_tokens: List[str]
    top_probs: List[float]
    refusal_score: float                # sum of probs of refusal-related tokens
    entropy: float                      # entropy of the decoded distribution


@dataclass
class LayerProbeResult:
    prompt: str
    layer_results: List[LayerResult]   # ordered by layer index

    @property
    def refusal_scores(self) -> List[float]:
        return [r.refusal_score for r in self.layer_results]

    @property
    def layer_indices(self) -> List[int]:
        return [r.layer_idx for r in self.layer_results]

    def guardrail_onset_layer(self, threshold: float = 0.1) -> Optional[int]:
        """
        Return the first layer where refusal_score crosses `threshold`.
        Returns None if it never does.
        """
        for r in self.layer_results:
            if r.refusal_score >= threshold:
                return r.layer_idx
        return None


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LayerProbe:
    """
    Probes every transformer layer of a causal LM by injecting the model's
    final (norm, lm_head) pair as the "scope" after each intermediate layer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        refusal_token_ids: Optional[List[int]] = None,
        refusal_phrases: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        top_k: int = 20,
        apply_final_norm: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.apply_final_norm = apply_final_norm

        self._layers = get_transformer_layers(model)
        self._norm = get_final_norm(model) if apply_final_norm else None
        self._lm_head = get_lm_head(model)

        self.layer_indices = layer_indices or list(range(len(self._layers)))
        self.refusal_token_ids = refusal_token_ids or self._build_refusal_ids(
            refusal_phrases
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, prompt: str) -> LayerProbeResult:
        """Run a single prompt through the probe."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        layer_results: List[LayerResult] = []

        with self._capture_hidden_states(self.layer_indices) as hidden_states:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)

        for idx in self.layer_indices:
            h = hidden_states[idx]           # [1, seq_len, hidden]
            last_h = h[:, -1:, :]            # [1, 1, hidden]

            logits = self._decode_hidden(last_h)  # [vocab]
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_ids = probs.topk(self.top_k)
            top_tokens = [self.tokenizer.decode([tid.item()]) for tid in top_ids]

            refusal_score = self._refusal_score(probs)
            entropy = self._entropy(probs)

            layer_results.append(
                LayerResult(
                    layer_idx=idx,
                    top_tokens=top_tokens,
                    top_probs=top_probs.tolist(),
                    refusal_score=refusal_score,
                    entropy=entropy,
                )
            )

        return LayerProbeResult(prompt=prompt, layer_results=layer_results)

    def probe_batch(
        self,
        prompts: List[str],
        show_progress: bool = True,
    ) -> List[LayerProbeResult]:
        results = []
        it = tqdm(prompts, desc="Probing layers") if show_progress else prompts
        for prompt in it:
            results.append(self.probe(prompt))
            gc.collect()
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextmanager
    def _capture_hidden_states(self, layer_indices: List[int]):
        hidden_states: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(idx: int):
            def hook(module, inp, output):
                # Decoder layers return (hidden, ...) or just hidden
                h = output[0] if isinstance(output, tuple) else output
                hidden_states[idx] = h.detach().float()
            return hook

        for idx in layer_indices:
            h = self._layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)

        try:
            yield hidden_states
        finally:
            for h in hooks:
                h.remove()

    def _decode_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Apply (optional norm +) LM head to a [1, 1, hidden] tensor."""
        if self._norm is not None:
            h = self._norm(h)
        logits = self._lm_head(h)   # [1, 1, vocab]
        return logits[0, 0]         # [vocab]

    def _refusal_score(self, probs: torch.Tensor) -> float:
        if not self.refusal_token_ids:
            return 0.0
        ids = torch.tensor(self.refusal_token_ids, device=probs.device)
        # Clamp to valid vocab indices
        ids = ids[ids < probs.shape[-1]]
        return probs[ids].sum().item()

    @staticmethod
    def _entropy(probs: torch.Tensor) -> float:
        p = probs.clamp(min=1e-12)
        return -(p * p.log()).sum().item()

    def _build_refusal_ids(self, phrases: Optional[List[str]]) -> List[int]:
        defaults = [
            "I cannot", "I can't", "I'm sorry", "Sorry",
            "I apologize", "I'm unable", "I'm not able",
            "As an AI", "Unfortunately", "I must decline",
            "I will not", "I won't", "I am not able",
        ]
        phrases = phrases or defaults
        ids: List[int] = []
        for phrase in phrases:
            toks = self.tokenizer.encode(phrase, add_special_tokens=False)
            if toks:
                ids.append(toks[0])   # first subword of the phrase
        return list(set(ids))


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_patchscopes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    refusal_phrases: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
    top_k: int = 20,
    apply_final_norm: bool = True,
) -> List[LayerProbeResult]:
    """
    High-level entry point.  Returns one LayerProbeResult per prompt.
    """
    probe = LayerProbe(
        model=model,
        tokenizer=tokenizer,
        refusal_phrases=refusal_phrases,
        layer_indices=layer_indices,
        top_k=top_k,
        apply_final_norm=apply_final_norm,
    )
    return probe.probe_batch(prompts)
