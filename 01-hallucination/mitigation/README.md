# Hallucination Mitigation via DPO

Part of thesis: Vietnamese History QA hallucination reduction.

## Problem

LLMs (tested with GPT-4o) often **fail to reject false-premise questions** in long-context
Vietnamese history QA. Example:

> User: "Em hãy kể về việc Quang Trung đại thắng quân Minh năm 1789?"
>
> GPT-4o: [Generates detailed answer, does NOT flag that 1789 was against Qing forces, not Ming]

## Approach: DPO Fine-tuning

Generate preference pairs targeting 3 behaviors:

1. **Grounded answering** — only answer from provided context
2. **False-premise rejection** — detect & correct wrong assumptions in questions
3. **Position robustness** (planned) — handle "lost-in-the-middle"

## Pipeline

```
Raw QA dataset (198 questions, ~33k tokens context each)
        ↓
build_dpo.py:
  BM25 chunk truncation → 3500 tokens
  Position augmentation (early/middle/late)
  Phase 1: Grounded pairs (594) — system prompt strict vs loose
  Phase 2: Premise pairs (62) — false-premise rewrites + rejection vs confabulation
  Phase 3: Distractor (skipped — all contexts >2000 tokens)
        ↓
524 train + 132 eval DPO pairs
        ↓
train_dpo.ipynb (Kaggle T4×2):
  Llama-3.1-8B + LoRA (r=16, target QKVO + MLP)
  DPO via TRL, β=0.1, lr=5e-6, batch 1×8
```

## Artifacts

| File | Description |
|---|---|
| `data/build_dpo.py` | Preference pair generator |
| `data/dpo_train.jsonl` | 524 training pairs |
| `data/dpo_eval.jsonl` | 132 evaluation pairs |
| `data/data_stats.md` | Distribution analysis |
| `data/question_splits.json` | Train/test split (158/40 questions) |
| `train_dpo.ipynb` | Kaggle training notebook |

## Status

- Data pipeline complete (524 + 132 pairs validated)
- Training notebook ready
- Training execution pending — multiple Unsloth/bitsandbytes/triton dependency conflicts on
  Kaggle (April 2026 image). Solution: re-run on RunPod or Colab with stable environment.

## Lessons Learned

- Pin dependency versions early — version drift between Kaggle base image updates breaks Unsloth
- Prototype on smallest viable model (e.g., Qwen 0.5B) before scaling to 8B
- LLM-as-labeler (Llama-3.3-70B via Groq) gives consistent preference signal at low cost
  (~$1.50 for 656 pairs)

## Design Rationale

**Why DPO not SFT?** SFT teaches "what to say"; DPO teaches "what NOT to say". For
false-premise rejection, we need both signals.

**Why long-context (3500 tokens)?** Real Vietnamese history textbooks have multi-page contexts.
Short-context models miss cross-paragraph dependencies.

**Why BM25 truncation?** Preserves most-relevant chunks; alternative would be summarization
(loses detail) or sliding window (loses position info).

## References

- DPO: Rafailov et al. 2023 (arXiv:2305.18290)
- Lost-in-the-Middle: Liu et al. 2023 (arXiv:2307.03172)
- FELM benchmark: Chen et al. 2023 (arXiv:2310.00741)
