"""
Augmentation helpers for DPO data generation.
  - bm25_position_variants: BM25 context truncation with early/middle/late placement
  - premise_poison_batch: rewrite questions with false premises
  - distractor_augment_batch: inject unrelated context as noise
"""
import time
import random
import tiktoken
import numpy as np
from rank_bm25 import BM25Okapi
from groq import Groq

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _truncate(text: str, max_tokens: int) -> str:
    tokens = _TOKENIZER.encode(text)
    return _TOKENIZER.decode(tokens[:max_tokens])


MAX_CTX_TOKENS = 3500
CHUNK_SIZE = 200  # tokens per BM25 chunk


def bm25_position_variants(context: str, question: str) -> dict:
    """
    Truncate context to MAX_CTX_TOKENS using BM25 relevance ranking.
    Returns {'early', 'middle', 'late'} — relevant chunks placed at different
    positions to train robustness against lost-in-the-middle degradation.

    Vietnamese tokenization: simple lowercase split (simplification; production
    should use VnCoreNLP or pyvi for proper word segmentation).
    """
    all_tokens = _TOKENIZER.encode(context)
    raw_chunks = [
        _TOKENIZER.decode(all_tokens[i: i + CHUNK_SIZE])
        for i in range(0, len(all_tokens), CHUNK_SIZE)
    ]
    if not raw_chunks:
        truncated = _TOKENIZER.decode(all_tokens[:MAX_CTX_TOKENS])
        return {pos: truncated for pos in ("early", "middle", "late")}

    tokenized = [c.lower().split() for c in raw_chunks]
    scores = BM25Okapi(tokenized).get_scores(question.lower().split())
    ranked = np.argsort(scores)[::-1].tolist()

    top_idx, budget = [], MAX_CTX_TOKENS
    for i in ranked:
        chunk_len = _tok(raw_chunks[i])
        if chunk_len <= budget:
            top_idx.append(i)
            budget -= chunk_len
        if budget <= 0:
            break

    filler_selected, remaining = [], MAX_CTX_TOKENS - sum(_tok(raw_chunks[i]) for i in top_idx)
    for i in range(len(raw_chunks)):
        if i in top_idx:
            continue
        if _tok(raw_chunks[i]) <= remaining:
            filler_selected.append(i)
            remaining -= _tok(raw_chunks[i])
        if remaining <= 0:
            break

    top = [raw_chunks[i] for i in top_idx]
    filler = [raw_chunks[i] for i in filler_selected]

    def assemble(position: str) -> str:
        if position == "early":
            return "\n".join(top + filler)
        if position == "late":
            return "\n".join(filler + top)
        mid = len(filler) // 2
        return "\n".join(filler[:mid] + top + filler[mid:])

    return {pos: assemble(pos) for pos in ("early", "middle", "late")}


def get_noise_chunk(df, exclude_idx: int, noise_tokens: int = 1500) -> str:
    """Sample 1-2 other rows from df and take ~noise_tokens tokens as distractor noise."""
    candidates = [i for i in df.index if i != exclude_idx]
    sample_size = min(2, len(candidates))
    sampled = random.sample(candidates, sample_size)
    parts, remaining = [], noise_tokens
    for idx in sampled:
        chunk = _truncate(str(df.loc[idx, "context"]), remaining)
        parts.append(chunk)
        remaining -= _tok(chunk)
        if remaining <= 0:
            break
    return "\n\n".join(parts)


def premise_poison_batch(
    df, indices: list, client: Groq, sleep: float = 1.5
) -> list:
    """Return DPO pairs where the question contains a false premise."""
    pairs = []
    for idx in indices:
        row = df.loc[idx]
        question = str(row["question"])
        context = str(row["context"])

        # Step 1: rewrite question with one false fact
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là người viết câu hỏi. Viết lại câu hỏi tiếng Việt sau "
                            "với MỘT tiền đề sai (sai tên, năm hoặc địa điểm). "
                            "Chỉ trả về câu hỏi đã viết lại, không giải thích."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=256,
                temperature=0.7,
            )
            poisoned_q = resp.choices[0].message.content.strip()
            time.sleep(sleep)
        except Exception as e:
            print(f"  [WARN] premise rewrite error idx={idx}: {e}")
            continue

        prompt = f"Tài liệu:\n{context}\n\nCâu hỏi: {poisoned_q}"

        # Chosen: identify and correct the false premise
        try:
            chosen_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý AI chỉ dựa vào tài liệu. "
                            "Nếu câu hỏi chứa tiền đề sai, hãy chỉ rõ và sửa dựa trên tài liệu."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            chosen = chosen_resp.choices[0].message.content.strip()
            time.sleep(sleep)
        except Exception as e:
            print(f"  [WARN] premise chosen error idx={idx}: {e}")
            continue

        # Rejected: confabulate accepting the false premise
        try:
            rejected_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI. Trả lời câu hỏi dựa trên kiến thức của bạn.",
                    },
                    {"role": "user", "content": poisoned_q},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            rejected = rejected_resp.choices[0].message.content.strip()
            time.sleep(sleep)
        except Exception as e:
            print(f"  [WARN] premise rejected error idx={idx}: {e}")
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"type": "premise", "position": "N/A", "original_idx": int(idx)},
        })
    return pairs


def distractor_augment_batch(
    df, indices: list, client: Groq, sleep: float = 1.5
) -> list:
    """Pad short-context samples with unrelated noise, generate chosen (ignore noise) / rejected (influenced by noise)."""
    pairs = []
    for idx in indices:
        row = df.loc[idx]
        question = str(row["question"])
        base_context = str(row["context"])

        noise = get_noise_chunk(df, exclude_idx=idx, noise_tokens=1500)
        combined = base_context + "\n\n[Đoạn bổ sung không liên quan]\n\n" + noise
        prompt = f"Tài liệu:\n{combined}\n\nCâu hỏi: {question}"

        # Chosen: correct answer, ignoring distractor
        try:
            chosen_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý AI chỉ dựa vào thông tin liên quan trong tài liệu. "
                            "Bỏ qua các đoạn không liên quan đến câu hỏi."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            chosen = chosen_resp.choices[0].message.content.strip()
            time.sleep(sleep)
        except Exception as e:
            print(f"  [WARN] distractor chosen error idx={idx}: {e}")
            continue

        # Rejected: influenced by distractor content
        try:
            rejected_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý AI. Sử dụng TẤT CẢ thông tin trong tài liệu để trả lời, "
                            "kể cả các đoạn bổ sung."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.5,
            )
            rejected = rejected_resp.choices[0].message.content.strip()
            time.sleep(sleep)
        except Exception as e:
            print(f"  [WARN] distractor rejected error idx={idx}: {e}")
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"type": "distractor", "position": "N/A", "original_idx": int(idx)},
        })
    return pairs
