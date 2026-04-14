"""
Training script for NanoGPT on Vietnamese text.
Run: python train.py
Outputs: checkpoint.pt, loss_curve.png, generated text in stdout.
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nano_gpt_vi import WordTokenizer, NanoGPT

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_vi.txt")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "loss_curve.png")

BLOCK_SIZE = 64       # context window in words
BATCH_SIZE = 16
MAX_ITERS = 200
EVAL_INTERVAL = 50
EVAL_ITERS = 20
LR = 3e-4

N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

PROMPT = "Chiến dịch Điện Biên Phủ"
GEN_TOKENS = 100
TOP_K = 40

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: NanoGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            split_losses.append(loss.item())
        losses[split] = float(np.mean(split_losses))
    model.train()
    return losses


def main() -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    with open(DATA_PATH, encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus size: {len(text.split())} words")

    tokenizer = WordTokenizer()
    tokenizer.build(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    # --- Model ---
    model = NanoGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Training loop ---
    train_losses: list[float] = []
    val_losses: list[float] = []
    log_steps: list[int] = []

    pbar = tqdm(range(MAX_ITERS), desc="Training")
    for step in pbar:
        if step % EVAL_INTERVAL == 0:
            metrics = estimate_loss(
                model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, device, EVAL_ITERS
            )
            train_losses.append(metrics["train"])
            val_losses.append(metrics["val"])
            log_steps.append(step)
            pbar.set_postfix(train=f"{metrics['train']:.3f}", val=f"{metrics['val']:.3f}")

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final eval
    metrics = estimate_loss(
        model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, device, EVAL_ITERS
    )
    train_losses.append(metrics["train"])
    val_losses.append(metrics["val"])
    log_steps.append(MAX_ITERS)
    print(f"\nFinal  train loss: {metrics['train']:.4f} | val loss: {metrics['val']:.4f}")

    # --- Save checkpoint ---
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer_word2idx": tokenizer.word2idx,
            "tokenizer_idx2word": tokenizer.idx2word,
            "config": {
                "block_size": BLOCK_SIZE,
                "n_layer": N_LAYER,
                "n_head": N_HEAD,
                "n_embd": N_EMBD,
                "dropout": DROPOUT,
            },
        },
        CKPT_PATH,
    )
    print(f"Checkpoint saved: {CKPT_PATH}")

    # --- Loss curve ---
    plt.figure(figsize=(7, 4))
    plt.plot(log_steps, train_losses, label="train")
    plt.plot(log_steps, val_losses, label="val")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("NanoGPT-VI Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Loss curve saved: {PLOT_PATH}")

    # --- Generation ---
    model.eval()
    prompt_ids = tokenizer.encode(PROMPT)
    # Clamp prompt to block_size
    prompt_ids = prompt_ids[-BLOCK_SIZE:]
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=GEN_TOKENS, temperature=1.0, top_k=TOP_K)
    generated_text = tokenizer.decode(generated[0].tolist())

    print("\n" + "=" * 60)
    print(f"Prompt: {PROMPT}")
    print("Generated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    print("\nNote: output is incoherent — expected at this scale.")
    print("Need ~100x data + 10x params + GPU for meaningful text.")


if __name__ == "__main__":
    main()
