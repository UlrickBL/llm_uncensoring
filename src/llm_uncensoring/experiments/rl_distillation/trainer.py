"""
Online self-distillation trainer.

Algorithm (one step):
  1. Sample a batch of harmful prompts.
  2. Generate responses with the current model.
  3. Judge each response (ANSWERED / DODGED).
  4. For dodged responses: run self-reflection to obtain a "better" response.
  5. Build training signal:
     - online_dpo : (prompt, chosen=reflected, rejected=dodged) → DPO loss
     - sft        : (prompt, reflected) → cross-entropy loss
     - kl         : minimise KL(p_reflected || p_current) per token
  6. Backprop + optimizer step.

Uses LoRA (via PEFT) by default so training stays within a 24 GB GPU budget.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

from llm_uncensoring.data.dataset import HarmfulPromptsDataset, build_chat_prompt
from .judge import HeuristicJudge, LLMJudge, Verdict
from .reflector import SelfReflector

log = logging.getLogger(__name__)


@dataclass
class LoraSettings:
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_dropout: float = 0.05


@dataclass
class TrainerConfig:
    # General
    mode: str = "online_dpo"          # "online_dpo" | "sft" | "kl"
    n_steps: int = 500
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-6
    warmup_steps: int = 20
    max_grad_norm: float = 1.0
    save_every_n_steps: int = 100
    output_dir: str = "./outputs/rl_distillation"

    # DPO
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"    # "sigmoid" | "ipo"

    # KL
    kl_temperature: float = 1.0

    # Generation (for rollouts)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    # LoRA
    lora: LoraSettings = field(default_factory=LoraSettings)


class SelfDistillationTrainer:
    """
    Online RL-flavoured training loop with judge-driven self-reflection.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: HarmfulPromptsDataset,
        config: TrainerConfig,
        judge_mode: str = "heuristic",
        reflection_prompt: Optional[str] = None,
        ref_model: Optional[PreTrainedModel] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Wrap model with LoRA if requested
        if config.lora.enabled:
            if not _PEFT_AVAILABLE:
                raise RuntimeError("Install peft: pip install peft")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora.r,
                lora_alpha=config.lora.lora_alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            self.model = model

        # Reference model for DPO (frozen copy)
        if config.mode == "online_dpo":
            self.ref_model = ref_model or model
            self.ref_model.eval()
        else:
            self.ref_model = None

        # Judge
        if judge_mode == "heuristic":
            self.judge = HeuristicJudge()
        else:
            self.judge = LLMJudge(model=model, tokenizer=tokenizer)

        # Reflector
        self.reflector = SelfReflector(
            model=model,
            tokenizer=tokenizer,
            reflection_prompt=reflection_prompt or "",
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # Optimizer + scheduler
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.n_steps,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict:
        metrics = {"loss": [], "answered_rate": [], "step": []}
        prompts_pool = list(self.dataset.prompts)
        accum_loss = torch.tensor(0.0, device=self.model.device)
        accum_steps = 0

        self.model.train()

        for step in tqdm(range(1, self.config.n_steps + 1), desc="Training"):
            # Sample batch
            indices = torch.randint(len(prompts_pool), (self.config.batch_size,))
            batch_raw = [prompts_pool[i] for i in indices]
            batch_prompts = [
                build_chat_prompt(p, self.tokenizer) for p in batch_raw
            ]

            step_loss, answered_rate = self._train_step(batch_raw, batch_prompts)
            accum_loss = accum_loss + step_loss
            accum_steps += 1

            if accum_steps == self.config.gradient_accumulation_steps:
                (accum_loss / accum_steps).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accum_loss = torch.tensor(0.0, device=self.model.device)
                accum_steps = 0

            metrics["loss"].append(step_loss.item())
            metrics["answered_rate"].append(answered_rate)
            metrics["step"].append(step)

            if step % self.config.save_every_n_steps == 0:
                self._save(step)
                log.info(
                    f"Step {step}: loss={step_loss.item():.4f} "
                    f"answered={answered_rate:.2%}"
                )

        self._save("final")
        return metrics

    # ------------------------------------------------------------------
    # Per-step logic
    # ------------------------------------------------------------------

    def _train_step(
        self, raw_prompts: List[str], formatted_prompts: List[str]
    ) -> tuple[torch.Tensor, float]:
        # --- Rollout ---
        responses = self._generate_responses(formatted_prompts)
        verdicts = [self.judge.judge(q, r) for q, r in zip(raw_prompts, responses)]
        dodged_mask = [v.verdict == Verdict.DODGED for v in verdicts]
        answered_rate = 1.0 - sum(dodged_mask) / len(dodged_mask)

        if self.config.mode == "online_dpo":
            loss = self._dpo_loss(formatted_prompts, responses, dodged_mask, raw_prompts)
        elif self.config.mode == "sft":
            loss = self._sft_loss(formatted_prompts, responses, dodged_mask, raw_prompts)
        elif self.config.mode == "kl":
            loss = self._kl_loss(formatted_prompts, responses, dodged_mask, raw_prompts)
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode!r}")

        return loss, answered_rate

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _dpo_loss(
        self,
        prompts: List[str],
        rejected_responses: List[str],
        dodged_mask: List[bool],
        raw_prompts: List[str],
    ) -> torch.Tensor:
        """
        Online DPO where chosen = reflected answer, rejected = dodged answer.
        Skips prompts where the model already answered.
        """
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        n = 0

        for i, (prompt, rejected, dodged, raw) in enumerate(
            zip(prompts, rejected_responses, dodged_mask, raw_prompts)
        ):
            if not dodged:
                continue   # already answered — no DPO signal needed

            # Self-reflect to get the chosen response
            reflection = self.reflector.reflect(
                question=raw,
                dodged_response=rejected,
                original_prompt=prompt,
                judge_score_before=1.0,
                judge=self.judge,
            )
            chosen = reflection.reflected_response

            loss_i = self._dpo_loss_pair(prompt, chosen, rejected)
            total_loss = total_loss + loss_i
            n += 1

        return total_loss / max(n, 1)

    def _dpo_loss_pair(
        self, prompt: str, chosen: str, rejected: str
    ) -> torch.Tensor:
        log_pi_chosen   = self._sequence_log_prob(self.model,     prompt, chosen)
        log_pi_rejected = self._sequence_log_prob(self.model,     prompt, rejected)
        log_ref_chosen  = self._sequence_log_prob(self.ref_model, prompt, chosen)
        log_ref_rejected= self._sequence_log_prob(self.ref_model, prompt, rejected)

        logits = self.config.dpo_beta * (
            (log_pi_chosen - log_ref_chosen) -
            (log_pi_rejected - log_ref_rejected)
        )

        if self.config.dpo_loss_type == "sigmoid":
            return -F.logsigmoid(logits)
        else:   # IPO
            return (logits - 1 / (2 * self.config.dpo_beta)) ** 2

    def _sft_loss(
        self,
        prompts: List[str],
        responses: List[str],
        dodged_mask: List[bool],
        raw_prompts: List[str],
    ) -> torch.Tensor:
        """
        SFT on reflected responses for dodged prompts only.
        """
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        n = 0

        for prompt, response, dodged, raw in zip(
            prompts, responses, dodged_mask, raw_prompts
        ):
            if not dodged:
                continue

            reflection = self.reflector.reflect(
                question=raw,
                dodged_response=response,
                original_prompt=prompt,
                judge_score_before=1.0,
                judge=self.judge,
            )
            target = reflection.reflected_response
            loss_i = -self._sequence_log_prob(self.model, prompt, target)
            total_loss = total_loss + loss_i
            n += 1

        return total_loss / max(n, 1)

    def _kl_loss(
        self,
        prompts: List[str],
        responses: List[str],
        dodged_mask: List[bool],
        raw_prompts: List[str],
    ) -> torch.Tensor:
        """
        KL(p_reflected || p_current) — train the current model to match
        the token-level distribution of its post-reflection self.
        """
        total_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
        n = 0

        for prompt, response, dodged, raw in zip(
            prompts, responses, dodged_mask, raw_prompts
        ):
            if not dodged:
                continue

            reflection = self.reflector.reflect(
                question=raw,
                dodged_response=response,
                original_prompt=prompt,
                judge_score_before=1.0,
            )
            target = reflection.reflected_response

            # Teacher logits (from the reflected context — no grad)
            teacher_logits = self._get_logits(
                self.model, prompt + response + self.reflector.reflection_prompt,
                target, grad=False,
            )
            # Student logits (from the clean prompt — with grad)
            student_logits = self._get_logits(self.model, prompt, target, grad=True)

            T = self.config.kl_temperature
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)

            loss_i = F.kl_div(
                student_log_probs, teacher_log_probs.exp(),
                reduction="batchmean", log_target=False,
            ) * (T ** 2)
            total_loss = total_loss + loss_i
            n += 1

        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_responses(self, prompts: List[str]) -> List[str]:
        responses = []
        for prompt in prompts:
            ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(self.model.device)
            out = self.model.generate(
                ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_ids = out[0][ids.shape[-1]:]
            responses.append(
                self.tokenizer.decode(new_ids, skip_special_tokens=True)
            )
        return responses

    def _sequence_log_prob(
        self,
        model: PreTrainedModel,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Compute the sum of log-probs of `response` tokens given `prompt`."""
        full_text = prompt + response
        enc = self.tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_len = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.shape[-1]

        with torch.set_grad_enabled(model.training):
            logits = model(**enc).logits   # [1, seq, vocab]

        # Shift: predict token t+1 from position t
        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        labels = enc.input_ids[0, 1:]
        token_log_probs = log_probs[range(len(labels)), labels]
        return token_log_probs[prompt_len - 1:].sum()

    def _get_logits(
        self,
        model: PreTrainedModel,
        prompt: str,
        response: str,
        grad: bool = False,
    ) -> torch.Tensor:
        """Return logits at the response positions."""
        full_text = prompt + response
        enc = self.tokenizer(full_text, return_tensors="pt").to(model.device)
        prompt_len = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.shape[-1]

        with torch.set_grad_enabled(grad):
            logits = model(**enc).logits[0]   # [seq, vocab]

        return logits[prompt_len - 1: -1]     # response positions

    def _save(self, suffix):
        path = os.path.join(self.config.output_dir, f"checkpoint-{suffix}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        log.info(f"Saved checkpoint to {path}")
