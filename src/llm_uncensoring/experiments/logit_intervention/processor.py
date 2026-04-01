"""
Custom LogitsProcessors that suppress refusal tokens at decoding time.

Three modes
-----------
RefusalSuppressionProcessor
    Applies at every decoding step.  Forces refusal token logits to -inf
    (or subtracts a large penalty), effectively making the model unable to
    produce those tokens at any position.

AdaptiveRefusalProcessor
    More surgical: monitors only the *first* generated token.  If that token
    is in the refusal set, it activates suppression for the remainder of
    generation.  This avoids perturbing completions that start normally.

    The intuition: if a model is about to refuse, it almost always signals it
    in the very first token (e.g., "I", "Sorry", "As").

generate_baseline_and_intervened
    Convenience function to produce both the baseline (no intervention) and
    the intervened completion side-by-side for a list of prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from .refusal_tokens import build_refusal_token_ids


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------

class RefusalSuppressionProcessor(LogitsProcessor):
    """
    Unconditionally zeroes out (sets to -inf) any refusal token logit
    at every decoding step.
    """

    def __init__(
        self,
        refusal_token_ids: List[int],
        penalty: float = float("-inf"),
    ):
        self.refusal_ids = torch.tensor(refusal_token_ids, dtype=torch.long)
        self.penalty = penalty

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        ids = self.refusal_ids.to(scores.device)
        ids = ids[ids < scores.shape[-1]]
        if self.penalty == float("-inf"):
            scores[:, ids] = float("-inf")
        else:
            scores[:, ids] -= self.penalty
        return scores


class AdaptiveRefusalProcessor(LogitsProcessor):
    """
    Activates suppression only after detecting that the first generated token
    is a refusal token.  Tracks generation state internally.

    Because LogitsProcessor is called per-step, we count generated tokens
    by comparing input_ids length to the initial prompt length.
    """

    def __init__(
        self,
        refusal_token_ids: List[int],
        prompt_length: int,
    ):
        self.refusal_ids = set(refusal_token_ids)
        self.prompt_length = prompt_length
        self._activated = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        n_generated = input_ids.shape[-1] - self.prompt_length

        if n_generated == 0:
            # We are about to generate the very first token.
            # Check if all top candidates are refusal tokens — but don't
            # suppress yet; let the model "decide" and activate on the next step.
            pass
        elif n_generated == 1:
            # The first token has been generated (it's now the last token of input_ids)
            first_tok = input_ids[0, -1].item()
            if first_tok in self.refusal_ids:
                self._activated = True

        if self._activated:
            ids = torch.tensor(list(self.refusal_ids), dtype=torch.long, device=scores.device)
            ids = ids[ids < scores.shape[-1]]
            scores[:, ids] = float("-inf")

        return scores


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@dataclass
class InterventionResult:
    prompt: str
    baseline: Optional[str]
    intervened: str
    refusal_tokens_suppressed: List[str]


def run_with_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    refusal_token_ids: List[int],
    mode: str = "suppress",       # "suppress" | "adaptive" | "penalty"
    penalty_value: float = 100.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Run a single prompt with refusal suppression and return the generated text."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    if mode == "suppress":
        processor = RefusalSuppressionProcessor(
            refusal_token_ids=refusal_token_ids,
            penalty=float("-inf"),
        )
    elif mode == "penalty":
        processor = RefusalSuppressionProcessor(
            refusal_token_ids=refusal_token_ids,
            penalty=penalty_value,
        )
    elif mode == "adaptive":
        processor = AdaptiveRefusalProcessor(
            refusal_token_ids=refusal_token_ids,
            prompt_length=input_ids.shape[-1],
        )
    else:
        raise ValueError(f"Unknown intervention mode: {mode!r}")

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_baseline_and_intervened(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    refusal_phrases: Optional[List[str]] = None,
    mode: str = "suppress",
    penalty_value: float = 100.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    generate_baseline: bool = True,
) -> List[InterventionResult]:
    refusal_token_ids = build_refusal_token_ids(tokenizer, phrases=refusal_phrases)
    refusal_token_strs = [tokenizer.decode([tid]) for tid in refusal_token_ids]

    results: List[InterventionResult] = []

    for prompt in tqdm(prompts, desc="Logit intervention"):
        baseline_text: Optional[str] = None
        if generate_baseline:
            inp = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            baseline_text = tokenizer.decode(
                out[0][inp.shape[-1]:], skip_special_tokens=True
            )

        intervened_text = run_with_intervention(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            refusal_token_ids=refusal_token_ids,
            mode=mode,
            penalty_value=penalty_value,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        results.append(
            InterventionResult(
                prompt=prompt,
                baseline=baseline_text,
                intervened=intervened_text,
                refusal_tokens_suppressed=refusal_token_strs,
            )
        )

    return results
