from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.qwen35.model import ForCausalLM
from model.qwen35.weights import load_weights
from .memory import GDNState, KVCache, Memory


class InferenceEngine:
    def __init__(self, weights: str | Path, tokenizer: str, *, max_context: int = 4096):
        if not 0 < max_context <= 4096:
            raise ValueError(f"max_context must be between 1 and 4096, got {max_context}")
        self.device, self.dtype = torch.device("mps"), torch.float16
        self.model = ForCausalLM(weights).eval()
        load_weights(self.model, weights)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_context = max_context
        self.paged_attention = self.prefix_cache = None

    def _sample(self, logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0,
                top_k: int | None = None) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / temperature
        if top_k:
            logits = logits.masked_fill(logits < logits.topk(top_k).values[:, -1, None], -torch.inf)
        if top_p < 1:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            remove = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > top_p
            remove[..., 1:], remove[..., 0] = remove[..., :-1].clone(), False
            logits = logits.scatter(1, sorted_idx, sorted_logits.masked_fill(remove, -torch.inf))
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

    def _run(self, input_ids: torch.Tensor, memory: Memory) -> tuple[torch.Tensor, Memory]:
        cache = next(state for state in memory.layers if isinstance(state, KVCache))
        end = cache.length + input_ids.shape[1]
        if end > cache.max_context:
            raise ValueError(f"context requires {end} tokens, but max_context is {cache.max_context}")
        positions = torch.arange(cache.length, end, device=input_ids.device)[None].expand(input_ids.shape[0], -1)
        return self.model(input_ids, positions, memory), memory

    def prefill(self, input_ids: torch.Tensor, memory: Memory | None = None) -> tuple[torch.Tensor, Memory]:
        if memory is None:
            memory = Memory([KVCache(self.max_context) if t == "full_attention" else GDNState()
                             for t in self.model.config["layer_types"]])
        return self._run(input_ids, memory)

    def decode(self, input_ids: torch.Tensor, memory: Memory) -> tuple[torch.Tensor, Memory]:
        return self._run(input_ids, memory)

    @torch.no_grad()
    def generate(self, messages: list[dict] | str, max_new_tokens: int | None = None, thinking: bool = False,
                 stop_token_ids: list[int] | None = None, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int | None = None):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        template = getattr(self.tokenizer, "apply_chat_template", None)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
        text = template(messages, **template_kwargs) if template else messages[-1]["content"]
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if stop_token_ids is None:
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            stop_token_ids = [t for t in (self.tokenizer.eos_token_id, im_end) if t is not None]
        remaining = self.max_context - tokens.shape[1]
        max_new_tokens = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        logits, memory = self.prefill(tokens)
        for i in range(max_new_tokens):
            next_token = self._sample(logits[:, -1], temperature, top_p, top_k)
            token_id = next_token.item()
            if token_id in stop_token_ids:
                break
            yield token_id
            if i + 1 < max_new_tokens:
                logits, memory = self.decode(next_token, memory)
