"""
EDA helper — generates data_stats.md and context_length_dist.png.
Called by build_dpo.py.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def tok(text: str) -> int:
    return len(_ENC.encode(str(text)))


def generate_eda(
    df: pd.DataFrame,
    out_dir: Path,
    train_id_set: set | None = None,
    test_id_set: set | None = None,
) -> None:
    stats_md = out_dir / "data_stats.md"
    stats_png = out_dir / "context_length_dist.png"

    ctx_lens = df["context"].apply(tok)
    q_lens = df["question"].apply(tok)

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ctx_lens, bins=30, color="steelblue", edgecolor="white")
    axes[0].set(title="Context Length Distribution (tokens)", xlabel="Tokens")
    axes[1].hist(q_lens, bins=20, color="coral", edgecolor="white")
    axes[1].set(title="Question Length Distribution (tokens)", xlabel="Tokens")
    plt.tight_layout()
    plt.savefig(stats_png, dpi=100)
    plt.close()

    samples = df.sample(3, random_state=42)
    sample_rows = "\n".join(
        f"- **id={r.id}** | `{str(r.question)[:80]}...` | ctx={tok(r.context)} tokens"
        for _, r in samples.iterrows()
    )

    md = f"""## Dataset Overview

| Field | Value |
|-------|-------|
| Samples | {len(df)} |
| Columns | {', '.join(df.columns)} |

> **Note:** Column `type` has values `"TRUE"`/`"FALSE"` with unclear semantic
> (possibly indicates question difficulty or source). It is **ignored** in data
> generation — all rows are treated uniformly.

## Context Length Distribution (tokens)

| Stat | Value |
|------|-------|
| Min | {int(ctx_lens.min())} |
| Max | {int(ctx_lens.max())} |
| Mean | {ctx_lens.mean():.0f} |
| Median | {int(ctx_lens.median())} |
| p95 | {ctx_lens.quantile(0.95):.0f} |

![Context length histogram](context_length_dist.png)

## Question Length Distribution (tokens)

| Stat | Value |
|------|-------|
| Min | {int(q_lens.min())} |
| Max | {int(q_lens.max())} |
| Mean | {q_lens.mean():.0f} |
| Median | {int(q_lens.median())} |
| p95 | {q_lens.quantile(0.95):.0f} |

## Sample Examples (3 random rows, truncated)

{sample_rows}
"""
    # Append split stats if split info provided
    if train_id_set is not None and test_id_set is not None:
        train_df = df[df["id"].astype(str).isin(train_id_set)]
        test_df  = df[df["id"].astype(str).isin(test_id_set)]
        split_section = f"""
## Train / Test Split (question-level, no leakage)

| Split | Questions | Expected pairs (grounded ×3 + premise + distractor) |
|-------|-----------|------------------------------------------------------|
| Train | {len(train_df)} | ~{len(train_df) * 3 + 50 + 24} |
| Test  | {len(test_df)} | ~{len(test_df) * 3 + 12 + 6} |

Split seed: 42. Saved to `question_splits.json`.
"""
        md += split_section

    stats_md.write_text(md, encoding="utf-8")
    print(f"[EDA] -> {stats_md.name}, {stats_png.name}")
