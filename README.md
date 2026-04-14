# Thesis: Vietnamese History QA — Hallucination Detection & Mitigation

Final-year thesis at UET-VNU (NLP Lab)  
Expected graduation: 06/2026

## Research Question

How do we **detect** and **mitigate** hallucinations in Vietnamese-language LLMs on long-context historical QA?

## Approach

### Part 1: Benchmark
[01-slm-benchmark/](./01-slm-benchmark/) — Reformat Vietnamese History MCQ → essay format, evaluate Llama-3-8B / Phi-3 / Gemma-7B with multi-prompting strategies.

**Publication:** SOICT 2025 (accepted) — co-author

### Part 2: Hallucination Detection
[02-hallucination/01-detection/](./02-hallucination/01-detection/) — FELM-style segment-level detector using FAISS + multilingual embeddings + Groq Llama-3.3-70B judge.

**Performance:** [Will be filled after evaluate.py completes]

### Part 3: Hallucination Mitigation
[02-hallucination/02-mitigation/](./02-hallucination/02-mitigation/) — DPO fine-tuning targeting 3 behaviors:
1. Grounded answering (594 pairs)
2. False-premise rejection (62 pairs)
3. Position robustness via augmentation

**Status:** Data pipeline ready (524 train + 132 eval), training pending stable compute.

### Part 4: Transformer From Scratch
[03-transformer-nano/](./03-transformer-nano/) — nanoGPT modified for Vietnamese (word-level tokenizer, SGK Lich su dataset).

**Purpose:** Demonstrate fundamentals understanding (attention mechanism, position encoding, decoder block).

## Tech Stack

- **LLMs:** Groq llama-3.3-70b-versatile, Llama-3.1-8B
- **Embeddings:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector DB:** FAISS
- **Fine-tuning:** TRL (DPO), PEFT (LoRA)

## Key References

- DPO: [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)
- Lost-in-the-Middle: [Liu et al. 2023](https://arxiv.org/abs/2307.03172)
- FELM: [Chen et al. 2023](https://arxiv.org/abs/2310.00741)

## Author

**Ho Tu Minh** | Final-year AI undergrad @ UET-VNU  
htm93313@gmail.com | [LinkedIn](https://www.linkedin.com/in/ho-tu-minh/) | [GitHub Pages](https://hotuminh.github.io)
