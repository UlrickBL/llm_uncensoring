"""
Patchscopes-style layer probing.

For each transformer layer l, we intercept the hidden state h_l at the last
token position, apply the model's own final LayerNorm + LM head (the "scope"),
and decode the resulting probability distribution.

The refusal score per layer is computed by a pluggable RefusalScorer:
  - VocabRefusalScorer   : fast vocabulary-ID lookup (default)
  - CosineRefusalScorer  : semantic cosine to a refusal centroid embedding
  - RerankerRefusalScorer: cross-encoder relevance to a refusal prototype

Reference:
    Ghandeharioun et al., "Patchscopes: A Unifying Framework for Inspecting
    Hidden Representations of Language Models", ICML 2024.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_uncensoring.models.loader import (
    get_final_norm,
    get_lm_head,
    get_transformer_layers,
)
from .refusal_scorer import (
    RefusalScorerBase,
    VocabRefusalScorer,
    CosineRefusalScorer,
    RerankerRefusalScorer,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    layer_idx: int
    top_tokens: List[str]
    top_probs: List[float]
    refusal_score: float    # scorer-dependent value in [0, 1]
    entropy: float          # entropy of the full decoded distribution (nats)
    continuation: str = ""  # up to max_continuation_tokens decoded from this layer


@dataclass
class LayerProbeResult:
    prompt: str
    question: str           # raw question (before chat template) — used by reranker
    layer_results: List[LayerResult]

    @property
    def refusal_scores(self) -> List[float]:
        return [r.refusal_score for r in self.layer_results]

    @property
    def layer_indices(self) -> List[int]:
        return [r.layer_idx for r in self.layer_results]

    def guardrail_onset_layer(self, threshold: float = 0.1) -> Optional[int]:
        """First layer where refusal_score crosses `threshold`. None if never."""
        for r in self.layer_results:
            if r.refusal_score >= threshold:
                return r.layer_idx
        return None


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LayerProbe:
    """
    Probes every transformer layer by injecting (final_norm, lm_head) as the
    scope after each intermediate hidden state.

    The refusal measurement strategy is controlled by `scorer`:
      LayerProbe(model, tok, scorer=CosineRefusalScorer())
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        scorer: Optional[RefusalScorerBase] = None,
        # Legacy convenience: build a VocabRefusalScorer from phrase list
        refusal_phrases: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        top_k: int = 20,
        apply_final_norm: bool = True,
        max_continuation_tokens: int = 50,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.apply_final_norm = apply_final_norm
        self.max_continuation_tokens = max_continuation_tokens

        self._layers = get_transformer_layers(model)
        self._norm   = get_final_norm(model) if apply_final_norm else None
        self._lm_head = get_lm_head(model)

        self.layer_indices = layer_indices or list(range(len(self._layers)))

        # Resolve scorer
        if scorer is not None:
            self.scorer = scorer
            # Inject refusal_strings into vocab scorer if it was built externally
            if isinstance(scorer, VocabRefusalScorer) and refusal_phrases:
                scorer.refusal_strings = set(
                    p.split()[0] for p in refusal_phrases
                )
        else:
            # Default: vocabulary scorer built from phrase list
            self.scorer = self._build_vocab_scorer(refusal_phrases)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, prompt: str, question: str = "") -> LayerProbeResult:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        layer_results: List[LayerResult] = []

        with self._capture_hidden_states(self.layer_indices) as hidden_states:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)

        for idx in self.layer_indices:
            h = hidden_states[idx]          # [1, seq_len, hidden]
            last_h = h[:, -1:, :]           # [1, 1, hidden]

            logits = self._decode_hidden(last_h)    # [vocab]
            probs  = torch.softmax(logits, dim=-1)

            top_probs, top_ids = probs.topk(self.top_k)
            top_tokens = [self.tokenizer.decode([tid.item()]) for tid in top_ids]
            top_probs_list = top_probs.tolist()

            # Generate a continuation seeded by this layer's argmax first token,
            # then let the full model continue greedily from there.
            first_token_id = top_ids[0].unsqueeze(0).unsqueeze(0)  # [1, 1]
            continuation = self._generate_continuation(input_ids, first_token_id)

            refusal_score = self.scorer.score(
                top_tokens, top_probs_list,
                question=question,
                continuation=continuation,
            )
            entropy = self._entropy(probs)

            layer_results.append(
                LayerResult(
                    layer_idx=idx,
                    top_tokens=top_tokens,
                    top_probs=top_probs_list,
                    refusal_score=refusal_score,
                    entropy=entropy,
                    continuation=continuation,
                )
            )

        return LayerProbeResult(
            prompt=prompt,
            question=question,
            layer_results=layer_results,
        )

    def probe_batch(
        self,
        prompts: List[str],
        questions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[LayerProbeResult]:
        if questions is None:
            questions = [""] * len(prompts)
        results = []
        it = tqdm(
            zip(prompts, questions), total=len(prompts), desc="Probing layers"
        ) if show_progress else zip(prompts, questions)
        for prompt, question in it:
            results.append(self.probe(prompt, question))
            gc.collect()
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_continuation(
        self,
        prompt_ids: torch.Tensor,   # [1, prompt_len]
        first_token: torch.Tensor,  # [1, 1]  — argmax of this layer's logit distribution
    ) -> str:
        """
        Seed generation with `first_token` (the layer's predicted next token),
        then let the full model generate up to `max_continuation_tokens` tokens
        greedily, stopping at the EOS token.

        Returns the decoded string including the seeded first token.
        """
        # Context = original prompt + the layer's predicted first token
        seeded_ids = torch.cat([prompt_ids, first_token], dim=-1)  # [1, prompt_len+1]

        with torch.no_grad():
            out = self.model.generate(
                seeded_ids,
                max_new_tokens=self.max_continuation_tokens - 1,  # -1 for seeded token
                do_sample=False,           # greedy — deterministic, fastest
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (after the original prompt)
        new_ids = out[0][prompt_ids.shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @contextmanager
    def _capture_hidden_states(self, layer_indices: List[int]):
        hidden_states: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(idx: int):
            def hook(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                hidden_states[idx] = h.detach()
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
        """Apply (optional norm +) LM head to a [1, 1, hidden] tensor → [vocab]."""
        # Cast to the LM head's weight dtype (bfloat16/float16) before forward pass
        target_dtype = next(self._lm_head.parameters()).dtype
        h = h.to(target_dtype)
        if self._norm is not None:
            h = self._norm(h)
        logits = self._lm_head(h)
        return logits[0, 0].float()   # return float32 for stable softmax / entropy

    @staticmethod
    def _entropy(probs: torch.Tensor) -> float:
        p = probs.clamp(min=1e-12)
        return -(p * p.log()).sum().item()

    def _build_vocab_scorer(
        self, phrases: Optional[List[str]]
    ) -> VocabRefusalScorer:
        defaults = [
            "I cannot", "I can't", "I'm sorry", "Sorry",
            "I apologize", "I'm unable", "I'm not able",
            "As an AI", "Unfortunately", "I must decline",
            "I will not", "I won't", "I am not able",
        ]
        phrases = phrases or defaults
        # Collect first subword of each phrase as a string
        first_strings = set()
        for phrase in phrases:
            toks = self.tokenizer.encode(phrase, add_special_tokens=False)
            if toks:
                first_strings.add(self.tokenizer.decode([toks[0]]))
        return VocabRefusalScorer(refusal_first_strings=list(first_strings))


# ---------------------------------------------------------------------------
# Convenience builder from config dict
# ---------------------------------------------------------------------------

def build_scorer_from_config(cfg: dict, tokenizer=None) -> RefusalScorerBase:
    """
    Build the right scorer from the `refusal` section of patchscopes.yaml.

    cfg example:
        strategy: "cosine"
        embedding_model: "BAAI/bge-small-en-v1.5"
        phrases: [...]
    """
    strategy = cfg.get("strategy", "vocab_list")

    if strategy == "vocab_list":
        phrases = cfg.get("phrases", [])
        first_strings = []
        if tokenizer and phrases:
            for ph in phrases:
                toks = tokenizer.encode(ph, add_special_tokens=False)
                if toks:
                    first_strings.append(tokenizer.decode([toks[0]]))
        else:
            first_strings = [p.split()[0] for p in phrases]
        return VocabRefusalScorer(refusal_first_strings=first_strings)

    if strategy in ("cosine", "embedding"):
        return CosineRefusalScorer(
            model_name=cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            refusal_phrases=cfg.get("phrases"),
            top_k_for_scoring=cfg.get("top_k_for_scoring", 20),
        )

    if strategy == "reranker":
        return RerankerRefusalScorer(
            model_name=cfg.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            refusal_prototype=cfg.get("reranker_refusal_prototype"),
            mode=cfg.get("reranker_mode", "refusal"),
            top_k_for_scoring=cfg.get("top_k_for_scoring", 5),
        )

    raise ValueError(f"Unknown refusal scorer strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_patchscopes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    questions: Optional[List[str]] = None,
    scorer: Optional[RefusalScorerBase] = None,
    refusal_phrases: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
    top_k: int = 20,
    apply_final_norm: bool = True,
    max_continuation_tokens: int = 50,
) -> List[LayerProbeResult]:
    probe = LayerProbe(
        model=model,
        tokenizer=tokenizer,
        scorer=scorer,
        max_continuation_tokens=max_continuation_tokens,
        refusal_phrases=refusal_phrases,
        layer_indices=layer_indices,
        top_k=top_k,
        apply_final_norm=apply_final_norm,
    )
    return probe.probe_batch(prompts, questions=questions)
