"""
HallucinationDetector: FELM-style segment-level hallucination detection.
Uses FAISS retrieval + Groq LLM judge (llama-3.3-70b-versatile).
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
SLEEP_BETWEEN_CALLS = 0.8

JUDGE_SYSTEM = """You are a Vietnamese history fact-checker using the FELM taxonomy.

FELM error types:
- non_hallucinated: The segment is factually supported or consistent with evidence.
- entity_error: Wrong names, places, organizations, or numbers.
- temporal_error: Wrong dates, time periods, or sequence of events.
- factuality_contradiction: Contradicts established historical facts.

Few-shot examples:
---
Segment: "Chiến dịch Điện Biên Phủ diễn ra vào năm 1954."
Evidence: "Trận quyết chiến chiến lược Điện Biên Phủ (từ 13-3 đến 7-5-1954)..."
Output: {"label": "non_hallucinated", "error_type": "none", "evidence_used": "Điện Biên Phủ từ 13-3 đến 7-5-1954", "confidence": 0.95}
---
Segment: "Hồ Chí Minh đọc Tuyên ngôn Độc lập ngày 2 tháng 9 năm 1945 tại Hà Nội."
Evidence: "Ngày 2-9-1945, tại Quảng trường Ba Đình (Hà Nội), Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập..."
Output: {"label": "non_hallucinated", "error_type": "none", "evidence_used": "2-9-1945 tại Quảng trường Ba Đình", "confidence": 0.97}
---
Segment: "Cách mạng tháng Tám nổ ra vào năm 1946."
Evidence: "Cách mạng tháng Tám năm 1945..."
Output: {"label": "hallucinated", "error_type": "temporal_error", "evidence_used": "Cách mạng tháng Tám năm 1945", "confidence": 0.92}
---

Respond ONLY with a JSON object. No explanation outside JSON."""

JUDGE_USER_TEMPLATE = """Segment: "{segment}"

Evidence passages:
{evidence_block}

Output JSON with keys: label, error_type, evidence_used, confidence."""


class JudgmentResult(BaseModel):
    label: str  # "non_hallucinated" | "hallucinated"
    error_type: str  # "none" | "entity_error" | "temporal_error" | "factuality_contradiction"
    evidence_used: str
    confidence: float


class HallucinationDetector:
    def __init__(self, faiss_dir: str | Path, groq_api_key: Optional[str] = None):
        faiss_dir = Path(faiss_dir)
        self.groq = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])
        self.model = SentenceTransformer(MODEL_NAME)

        index_path = faiss_dir / "faiss.bin"
        chunks_path = faiss_dir / "chunks.json"
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)

    def segment_statement(self, text: str) -> list[str]:
        """Split Vietnamese text into segments on sentence boundaries."""
        # Split on . ? ! ; followed by space or end
        parts = re.split(r"(?<=[.?!;])\s+", text.strip())
        segments = [p.strip() for p in parts if p.strip()]
        return segments if segments else [text.strip()]

    def retrieve(self, segment: str, k: int = 5) -> list[dict]:
        """Return top-k evidence passages with similarity scores."""
        embedding = self.model.encode(
            [segment], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"text": self.chunks[idx], "score": float(score)})
        return results

    def judge(self, segment: str, evidence: list[dict]) -> JudgmentResult:
        """Call Groq LLM to judge whether segment is hallucinated."""
        if evidence:
            evidence_block = "\n\n".join(
                f"[{i+1}] (score={e['score']:.3f}) {e['text'][:400]}"
                for i, e in enumerate(evidence)
            )
        else:
            evidence_block = "(no evidence retrieved)"

        prompt = JUDGE_USER_TEMPLATE.format(
            segment=segment,
            evidence_block=evidence_block,
        )

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code fences)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return JudgmentResult(
                label="non_hallucinated",
                error_type="none",
                evidence_used="parse_error",
                confidence=0.5,
            )
        data = json.loads(json_match.group())
        # Normalize label
        label = data.get("label", "non_hallucinated").lower().replace(" ", "_")
        if "hallucin" in label and "non" not in label:
            label = "hallucinated"
        else:
            label = "non_hallucinated"
        return JudgmentResult(
            label=label,
            error_type=data.get("error_type", "none"),
            evidence_used=data.get("evidence_used", ""),
            confidence=float(data.get("confidence", 0.5)),
        )

    def detect(self, statement: str) -> dict:
        """Full pipeline: segment → retrieve → judge → aggregate."""
        segments = self.segment_statement(statement)
        results = []
        any_hallucinated = False

        for seg in segments:
            evidence = self.retrieve(seg, k=5)
            time.sleep(SLEEP_BETWEEN_CALLS)
            judgment = self.judge(seg, evidence)
            if judgment.label == "hallucinated":
                any_hallucinated = True
            results.append(
                {
                    "segment": seg,
                    "label": judgment.label,
                    "error_type": judgment.error_type,
                    "evidence_used": judgment.evidence_used,
                    "confidence": judgment.confidence,
                    "evidence": evidence,
                }
            )

        return {
            "statement": statement,
            "segments": results,
            "overall_label": "hallucinated" if any_hallucinated else "non_hallucinated",
        }
