"""
Minimal GPT implementation for Vietnamese text generation.
Based on Karpathy's nanoGPT: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

Changes vs original:
- Word-level tokenizer (not char-level): Vietnamese diacritics make char-level inefficient.
  A char-level vocab on Vietnamese would be ~200+ chars and encode sequences ~5x longer.
- Smaller architecture: 4 layers, 4 heads, n_embd=128 for CPU training.
- block_size=64 words instead of chars.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Word-level tokenizer
# ---------------------------------------------------------------------------

class WordTokenizer:
    """
    Builds vocabulary from whitespace-split tokens.
    <UNK> for out-of-vocab words at inference time.
    """

    UNK = "<UNK>"
    PAD = "<PAD>"

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def build(self, text: str) -> None:
        words = text.split()
        vocab = sorted(set(words))
        special = [self.PAD, self.UNK]
        self.idx2word = special + vocab
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def encode(self, text: str) -> list[int]:
        unk_id = self.word2idx[self.UNK]
        return [self.word2idx.get(w, unk_id) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.idx2word[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask: lower-triangular ones
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer decoder block: LayerNorm -> Attention -> LayerNorm -> MLP."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """
    Decoder-only GPT with positional embeddings.

    Architecture:
        Token Embedding + Positional Embedding
        -> N x (LayerNorm + MultiHeadAttn + LayerNorm + MLP)
        -> LayerNorm
        -> Linear (head)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: share embedding and output projection weights
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with optional top-k sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
