# 01-transformer-nano — MicroGPT from Scratch (Vietnamese)

A minimal decoder-only Transformer trained on Vietnamese historical text, built from scratch
following Karpathy's nanoGPT recipe.

## Architecture

```
Input tokens (word IDs)
        │
   ┌────┴────┐
   Token Emb  Position Emb   ← learned, block_size=64
   └────┬────┘
        │  (sum, dropout)
   ┌────▼────────────────────────────────┐
   │  Block × 4                          │
   │  ┌─────────────────────────────┐    │
   │  │  LayerNorm                  │    │
   │  │  Multi-Head Self-Attention  │ ←─ causal mask
   │  │  (4 heads, head_dim=32)     │    │
   │  │  + residual                 │    │
   │  ├─────────────────────────────┤    │
   │  │  LayerNorm                  │    │
   │  │  MLP: Linear→GELU→Linear    │    │
   │  │  (hidden = 4 × n_embd=512) │    │
   │  │  + residual                 │    │
   │  └─────────────────────────────┘    │
   └─────────────────────────────────────┘
        │
   LayerNorm → Linear (vocab_size)
        │
   Next-word logits
```

## Tokenizer choice: word-level vs alternatives

| Approach | Pros | Cons |
|---|---|---|
| **Word-level (used)** | Short sequences, natural for Vietnamese compound words | Large vocab (~8k words), OOV at inference |
| Char-level | Zero OOV, tiny vocab | Vietnamese has ~150 base chars + tone combos → sequences 5–8× longer, attention O(T²) explodes |
| BPE / SentencePiece | Best balance (used by LLaMA, GPT-4) | Needs ~10k+ lines of text to train meaningfully; overkill for demo |

Word-level is chosen here because the corpus is small (~7k unique words) and sequences must fit
in `block_size=64` for CPU training.

## Hyperparameters

| Param | Value |
|---|---|
| Layers | 4 |
| Heads | 4 |
| Embedding dim | 128 |
| Block size | 64 words |
| Batch size | 16 |
| Max iters | 200 |
| LR | 3e-4 (AdamW) |

## How to run

```bash
cd 01-transformer-nano
uv venv
uv pip install -r requirements.txt
python train.py
```

Outputs: `checkpoint.pt`, `loss_curve.png`, generated text in stdout.

## Reference

- Karpathy nanoGPT gist: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- Data: Vietnamese history textbook excerpt (~50 KB, ~7k tokens)

## Scope & Limitations

- **Output is incoherent.** At this scale (~300k params, ~7k training tokens, 200 iters) the
  model memorizes fragments but cannot generalize.
- **Scaling rule of thumb:** to produce meaningful text, need roughly 100× more training data
  (5 MB+), 10× more parameters (3M+), and a GPU.
- **OOV problem:** any word not in training data maps to `<UNK>` at inference.
- **No subword segmentation:** Vietnamese tonal diacritics are preserved but treated as opaque
  word units — the model has no morphological awareness.
- This is a from-scratch educational demo, not a production model.
