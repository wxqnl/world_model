"""Frozen CLIP text encoder wrapper for task-string conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPText(nn.Module):
    def __init__(self, hf_id: str, max_length: int = 32):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(hf_id)
        self.model = CLIPTextModel.from_pretrained(hf_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_length = max_length

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        out = self.model(**tok)
        # pooled output (BOS token after final layernorm) -> [B, D]
        return out.pooler_output
