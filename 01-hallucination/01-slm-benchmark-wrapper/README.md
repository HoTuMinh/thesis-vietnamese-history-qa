# Vietnamese History QA — SLM Benchmark Framework

> Part of my thesis at UET-VNU. Co-author SOICT 2025 (paper accepted).
> Full implementation: https://github.com/HoTuMinh/Framework-for-evaluating-SLMs-in-History-QA

## Problem

Vietnamese history education materials (mainly THPT-level) are predominantly multiple-choice.
Yet SLMs (Llama-3-8B, Phi-3) are designed for **generative** answering. Standard benchmarks
underestimate their reasoning capability when forced into MCQ format.

## My Approach

### Step 1: Format conversion

- Source: THPT National Exam history MCQ dataset
- Reformat: MCQ → essay (open-ended generative)
- Preserve: ground-truth answer + reasoning chain

### Step 2: Multi-strategy prompting

Tested 4 strategies on Llama-3-8B and Phi-3:

- Zero-shot
- Few-shot (3 examples)
- Chain-of-Thought (CoT)
- CoT + Few-shot combined

### Step 3: LLM-as-Judge evaluation

First attempt: rubric-based scoring (1–5 with descriptions)

- **Finding: Highly unstable.** Same answer scored differently across runs.
- Pivot: switch to **quantitative metrics** (truthfulness binary labels per claim,
  similar to TruthfulQA)

## Key Insights

1. **Format matters**: Llama-3-8B essay-format accuracy 23% higher than forced-MCQ accuracy
2. **CoT helps**: +12% accuracy on reasoning questions
3. **Judge calibration is hard**: Rubric scoring std-dev 1.4/5.0 across 3 runs of same answer

## Tech Stack

- Models: Llama-3-8B, Phi-3-mini, Gemma-7B
- Judge: GPT-4 + custom calibration
- Eval: TruthfulQA-style binary labels per claim

## Code

Full repo: https://github.com/HoTuMinh/Framework-for-evaluating-SLMs-in-History-QA

## Publication

SOICT 2025 (accepted) — "Evaluation Framework for Small Language Models on Vietnamese
Historical QA"
