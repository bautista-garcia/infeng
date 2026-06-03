from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.qwen35.model import ForCausalLM
from model.qwen35.weights import load_weights


class InferenceEngine:
    def __init__(
        self,
        config: str | Path,
        weights: str | Path,
        tokenizer: str,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
    ):
        self.device = torch.device(
            "mps"
            if device == "auto" and torch.backends.mps.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.dtype = (
            getattr(torch, dtype)
            if isinstance(dtype, str) and dtype != "auto"
            else dtype
        )
        if self.dtype == "auto":
            self.dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        self.model = ForCausalLM.build(
            config, device=self.device, dtype=self.dtype
        ).eval()
        self.report = load_weights(self.model, weights)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.paged_attention = self.prefix_cache = None

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            logits = logits.masked_fill(
                logits < logits.topk(top_k).values[:, -1, None], -torch.inf
            )
        if top_p < 1:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            remove = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p
            remove[..., 1:], remove[..., 0] = remove[..., :-1].clone(), False
            logits = logits.scatter(
                1, sorted_idx, sorted_logits.masked_fill(remove, -torch.inf)
            )
        return torch.multinomial(F.softmax(logits, dim=-1).cpu(), 1).to(logits.device)

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict] | str,
        max_new_tokens: int | None = None,
        thinking: bool = False,
        stop_token_ids: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            if hasattr(self.tokenizer, "apply_chat_template")
            else messages[-1]["content"]
        )
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        stop_token_ids = (
            [
                t
                for t in (
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                )
                if t is not None
            ]
            if stop_token_ids is None
            else stop_token_ids
        )
        max_new_tokens = (
            self.model.config.get("max_position_embeddings", 4096) - tokens.shape[1]
            if max_new_tokens is None
            else max_new_tokens
        )
        cache = None
        for _ in range(max_new_tokens):
            logits, cache = self.model(
                tokens[:, -1:] if cache else tokens, cache=cache, use_cache=True
            )
            next_token = self._sample(logits[:, -1].float(), temperature, top_p, top_k)
            tokens = torch.cat((tokens, next_token), dim=1)
            token_id = next_token.item()
            if token_id in stop_token_ids:
                break
            yield token_id
