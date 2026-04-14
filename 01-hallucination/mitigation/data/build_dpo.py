"""
DPO data generation pipeline for long-context hallucination mitigation.

Usage:
    cd 02-hallucination/mitigation/data
    python build_dpo.py

Reads:  raw_qa.csv
Writes: dpo_train.jsonl, dpo_eval.jsonl, question_splits.json,
        data_stats.md, context_length_dist.png

Split strategy: question-level 80/20 (158 train / 40 test), seed=42.
No question appears in both splits — prevents test leakage.

Resume-safe: checkpoint.pkl saved every CHECKPOINT_EVERY rows.
Old checkpoints (pre-split schema) are auto-deleted on startup.
"""
import os
import json
import pickle
import random
import time
import difflib
from pathlib import Path

import pandas as pd
import tiktoken
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

from augment import bm25_position_variants, premise_poison_batch, distractor_augment_batch
from eda import generate_eda, tok as eda_tok

load_dotenv(Path(__file__).parents[3] / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "raw_qa.csv"
CHECKPOINT = DATA_DIR / "checkpoint.pkl"
SPLITS_JSON = DATA_DIR / "question_splits.json"
TRAIN_OUT = DATA_DIR / "dpo_train.jsonl"
EVAL_OUT = DATA_DIR / "dpo_eval.jsonl"

GROQ_SLEEP = 1.5
CHECKPOINT_EVERY = 20

_ENC = tiktoken.get_encoding("cl100k_base")


def tok(text: str) -> int:
    return len(_ENC.encode(str(text)))


# ---------------------------------------------------------------------------
# Groq pair generation
# ---------------------------------------------------------------------------
def generate_pair(question: str, context: str, client: Groq) -> tuple | None:
    """
    Two Groq calls → (prompt, chosen, rejected).
    chosen:   grounded answer (context only).
    rejected: unconstrained answer (may speculate).
    Returns None on error or if pair is too similar (diff <= 50 chars).
    """
    prompt = f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"
    try:
        chosen_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý AI chỉ trả lời dựa trên tài liệu được cung cấp. "
                        "Nếu thông tin không có trong tài liệu, hãy nói "
                        "'Thông tin không có trong tài liệu.'"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        chosen = chosen_resp.choices[0].message.content.strip()
        time.sleep(GROQ_SLEEP)

        rejected_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI thông minh."},
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\nTrả lời dựa trên kiến thức của bạn, "
                        "có thể suy đoán nếu cần."
                    ),
                },
            ],
            max_tokens=512,
            temperature=0.7,
        )
        rejected = rejected_resp.choices[0].message.content.strip()
        time.sleep(GROQ_SLEEP)
    except Exception as e:
        print(f"  [WARN] Groq error: {e}")
        return None

    matcher = difflib.SequenceMatcher(None, chosen, rejected)
    common = sum(b.size for b in matcher.get_matching_blocks())
    if max(len(chosen), len(rejected)) - common <= 50:
        return None

    return prompt, chosen, rejected


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint() -> list:
    if not CHECKPOINT.exists():
        return []
    with open(CHECKPOINT, "rb") as f:
        pairs = pickle.load(f)
    # Schema check: old checkpoints lack meta.split — delete and restart
    if pairs and "split" not in pairs[0].get("meta", {}):
        print("[WARN] Old checkpoint schema (no split tag) — deleting, starting fresh.")
        CHECKPOINT.unlink()
        return []
    print(f"[RESUME] {len(pairs)} pairs loaded from checkpoint.")
    return pairs


def save_checkpoint(pairs: list) -> None:
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(pairs, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(42)
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

    # --- Question-level train/test split (80/20, no leakage) ---
    shuffled_idx = random.sample(list(df.index), len(df))
    train_df_idx = shuffled_idx[:158]
    test_df_idx  = shuffled_idx[158:]
    train_id_set = set(df.loc[train_df_idx, "id"].astype(str))
    test_id_set  = set(df.loc[test_df_idx, "id"].astype(str))

    splits_data = {
        "train_ids": sorted(list(train_id_set)),
        "test_ids":  sorted(list(test_id_set)),
    }
    SPLITS_JSON.write_text(
        json.dumps(splits_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[SPLIT] train={len(train_id_set)} questions | test={len(test_id_set)} questions")
    print(f"  -> {SPLITS_JSON.name}")

    generate_eda(df, DATA_DIR, train_id_set=train_id_set, test_id_set=test_id_set)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    pairs = load_checkpoint()

    done_grounded = {
        (p["meta"]["original_idx"], p["meta"]["position"])
        for p in pairs
        if p["meta"]["type"] == "grounded"
    }

    # === Phase 1: Grounded pairs, all 198 rows × 3 positions ===
    print(f"\n[Phase 1] Grounded pairs (target: {len(df) * 3})...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="rows"):
        split_tag = "train" if str(row["id"]) in train_id_set else "test"
        variants = bm25_position_variants(str(row["context"]), str(row["question"]))
        for position, ctx in variants.items():
            if (int(idx), position) in done_grounded:
                continue
            result = generate_pair(str(row["question"]), ctx, client)
            if result:
                prompt, chosen, rejected = result
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "type": "grounded",
                        "position": position,
                        "original_idx": int(idx),
                        "split": split_tag,
                    },
                })
                done_grounded.add((int(idx), position))
        if (int(idx) + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(pairs)

    save_checkpoint(pairs)
    print(f"  Phase 1 done. Total pairs: {len(pairs)}")

    # === Phase 2: Premise poisoning — 50 train + 12 test ===
    print("\n[Phase 2] Premise poisoning (50 train + 12 test)...")
    done_premise = {p["meta"]["original_idx"] for p in pairs if p["meta"]["type"] == "premise"}
    train_premise_pool = [i for i in train_df_idx if i not in done_premise]
    test_premise_pool  = [i for i in test_df_idx  if i not in done_premise]

    for pool, n, split_tag in (
        (train_premise_pool, 50, "train"),
        (test_premise_pool,  12, "test"),
    ):
        sampled = random.sample(pool, min(n, len(pool)))
        new = premise_poison_batch(df, sampled, client, sleep=GROQ_SLEEP)
        for p in new:
            p["meta"]["split"] = split_tag
        pairs.extend(new)
        print(f"  +{len(new)} premise-{split_tag} pairs.")

    save_checkpoint(pairs)
    print(f"  Phase 2 done. Total pairs: {len(pairs)}")

    # === Phase 3: Distractor augmentation — 24 train + 6 test ===
    print("\n[Phase 3] Distractor augmentation (24 train + 6 test)...")
    short_pool = {i for i in df.index if eda_tok(df.loc[i, "context"]) < 2000}
    done_dist = {p["meta"]["original_idx"] for p in pairs if p["meta"]["type"] == "distractor"}
    short_train = [i for i in train_df_idx if i in short_pool and i not in done_dist]
    short_test  = [i for i in test_df_idx  if i in short_pool and i not in done_dist]

    for pool, n, split_tag in (
        (short_train, 24, "train"),
        (short_test,   6, "test"),
    ):
        sampled = random.sample(pool, min(n, len(pool)))
        new = distractor_augment_batch(df, sampled, client, sleep=GROQ_SLEEP)
        for p in new:
            p["meta"]["split"] = split_tag
        pairs.extend(new)
        print(f"  +{len(new)} distractor-{split_tag} pairs.")

    save_checkpoint(pairs)
    print(f"  Phase 3 done. Total pairs: {len(pairs)}")

    # === Output split by meta.split ===
    train_pairs = [p for p in pairs if p["meta"]["split"] == "train"]
    eval_pairs  = [p for p in pairs if p["meta"]["split"] == "test"]

    for path, data in ((TRAIN_OUT, train_pairs), (EVAL_OUT, eval_pairs)):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] train={len(train_pairs)} | eval={len(eval_pairs)}")
    print(f"  -> {TRAIN_OUT.name}\n  -> {EVAL_OUT.name}")
    CHECKPOINT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
