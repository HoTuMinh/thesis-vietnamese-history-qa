"""
Build FAISS index from Vietnamese history textbook corpus.
Chunks text by paragraph with ~500 char target, 50 char overlap.
Embeds with paraphrase-multilingual-MiniLM-L12-v2.
Saves: index/faiss.bin, index/chunks.json
"""
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORPUS_PATH = Path(__file__).parent / "corpus" / "history_textbook.txt"
INDEX_DIR = Path(__file__).parent / "index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET_CHUNK_CHARS = 500
OVERLAP_CHARS = 50


def load_corpus(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, target: int = TARGET_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    # Split on blank lines first (paragraph boundaries)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # If adding this paragraph stays within target, accumulate
        if len(current) + len(para) + 1 <= target:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph is too long, split by sentences
            if len(para) > target:
                sentences = re.split(r"(?<=[.!?;])\s+", para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= target:
                        sub = (sub + " " + sent).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        # Overlap: take last `overlap` chars as prefix
                        prefix = sub[-overlap:] if len(sub) > overlap else sub
                        sub = (prefix + " " + sent).strip()
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                # Start new chunk with overlap from previous
                prefix = current[-overlap:] if len(current) > overlap else current
                current = (prefix + "\n" + para).strip()

    if current:
        chunks.append(current)

    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main():
    INDEX_DIR.mkdir(exist_ok=True)

    print(f"Loading corpus from {CORPUS_PATH} ...")
    text = load_corpus(CORPUS_PATH)
    print(f"Corpus length: {len(text):,} chars")

    print("Chunking ...")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks ...")
    embeddings = model.encode(
        chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}")

    print("Building FAISS index ...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss_path = INDEX_DIR / "faiss.bin"
    chunks_path = INDEX_DIR / "chunks.json"

    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index -> {faiss_path}")
    print(f"Saved chunks      -> {chunks_path}")
    print(f"Index size: {faiss_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
