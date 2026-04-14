"""
Smoke test — run before build_dpo.py full pipeline.
Tests: tiktoken, BM25 position augmentation, Groq API (4 calls total).
No files are written.

Usage:
    cd 02-hallucination/mitigation/data
    python test_smoke.py
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from augment import bm25_position_variants
from eda import tok
from build_dpo import generate_pair

load_dotenv(Path(__file__).parents[3] / ".env")


def main() -> None:
    df = pd.read_csv(Path(__file__).parent / "raw_qa.csv")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    for i, (idx, row) in enumerate(df.head(2).iterrows()):
        question = str(row["question"])
        context = str(row["context"])

        print(f"\n{'='*60}")
        print(f"Row {i+1} | id={row['id']}")
        print(f"Question: {question[:100]}...")
        print(f"Context tokens (original): {tok(context):,}")

        # BM25 truncation — all 3 positions
        variants = bm25_position_variants(context, question)
        for pos, ctx in variants.items():
            print(f"  [{pos:6}] tokens after truncate: {tok(ctx):,}")

        # generate_pair with 'early' variant (2 Groq calls)
        result = generate_pair(question, variants["early"], client)
        if result is None:
            print("  [WARN] generate_pair returned None (pair too similar or Groq error)")
        else:
            prompt, chosen, rejected = result
            print(f"  Chosen  ({len(chosen):4d} chars): {chosen[:200]}...")
            print(f"  Rejected({len(rejected):4d} chars): {rejected[:200]}...")

    print(f"\n{'='*60}")
    print("Smoke test done.")


if __name__ == "__main__":
    main()
