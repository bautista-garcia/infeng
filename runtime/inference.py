from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.qwen35.model import ForCausalLM
from model.qwen35.weights import load_weights


class InferenceEngine:
    def __init__(self, weights: str | Path, tokenizer: str):
        self.device, self.dtype = torch.device("mps"), torch.float16
        self.model = ForCausalLM(weights).eval()
        self.report = load_weights(self.model, weights)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
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
        max_position = self.model.config.get("max_position_embeddings", 4096)
        max_new_tokens = max_position - tokens.shape[1] if max_new_tokens is None else max_new_tokens
        cache, input_ids = None, tokens
        for _ in range(max_new_tokens):
            logits, cache = self.model(input_ids, cache=cache, use_cache=True)
            next_token = self._sample(logits[:, -1], temperature, top_p, top_k)
            input_ids = next_token
            token_id = next_token.item()
            if token_id in stop_token_ids:
                break
            yield token_id
