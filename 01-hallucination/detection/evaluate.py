"""
Evaluate HallucinationDetector on 100-sample subset of felm_vi_history.csv.
Computes segment-level + statement-level metrics.
Runs ablation: with RAG vs no RAG.
Saves: results/metrics.json, results/confusion_matrix.png, results/ablation_comparison.png
"""
import json
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from detector import HallucinationDetector

load_dotenv()

BASE_DIR = Path(__file__).parent
BENCHMARK_PATH = BASE_DIR / "benchmark" / "felm_vi_history.csv"
INDEX_DIR = BASE_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"
CACHE_PATH = BASE_DIR / "index" / "eval_cache.pkl"
N_SAMPLES = 100
RANDOM_STATE = 42


def load_benchmark(n: int = N_SAMPLES) -> pd.DataFrame:
    df = pd.read_csv(BENCHMARK_PATH)
    df = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def parse_json_col(val: str) -> list:
    """Parse a stringified JSON list column."""
    if isinstance(val, list):
        return val
    try:
        return json.loads(val)
    except Exception:
        return []


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def run_evaluation(detector: HallucinationDetector, df: pd.DataFrame, use_rag: bool = True) -> dict:
    """
    For each row, judge each pre-segmented segment from the CSV.
    Returns segment-level and statement-level predictions vs ground truth.
    """
    cache = load_cache()
    cache_key_prefix = "rag" if use_rag else "norag"

    seg_gt: list[str] = []
    seg_pred: list[str] = []
    seg_error_types: list[str] = []

    stmt_gt: list[str] = []
    stmt_pred: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating (rag={use_rag})"):
        segments = parse_json_col(row["segment"])
        labels = parse_json_col(row["label"])

        if not segments or not labels:
            continue

        # Statement-level ground truth: hallucinated if any segment is false
        stmt_has_hall = any(str(l).lower() == "false" for l in labels)
        stmt_gt.append("hallucinated" if stmt_has_hall else "non_hallucinated")

        stmt_pred_hall = False
        for seg, lbl in zip(segments, labels):
            gt_label = "hallucinated" if str(lbl).lower() == "false" else "non_hallucinated"
            seg_gt.append(gt_label)

            cache_key = f"{cache_key_prefix}::{seg}"
            if cache_key in cache:
                judgment = cache[cache_key]
            else:
                if use_rag:
                    evidence = detector.retrieve(seg, k=5)
                else:
                    evidence = []
                time.sleep(0.8)
                judgment = detector.judge(seg, evidence)
                cache[cache_key] = judgment
                save_cache(cache)

            seg_pred.append(judgment.label)
            seg_error_types.append(judgment.error_type)
            if judgment.label == "hallucinated":
                stmt_pred_hall = True

        stmt_pred.append("hallucinated" if stmt_pred_hall else "non_hallucinated")

    return {
        "seg_gt": seg_gt,
        "seg_pred": seg_pred,
        "seg_error_types": seg_error_types,
        "stmt_gt": stmt_gt,
        "stmt_pred": stmt_pred,
    }


def compute_metrics(results: dict) -> dict:
    seg_gt = results["seg_gt"]
    seg_pred = results["seg_pred"]
    stmt_gt = results["stmt_gt"]
    stmt_pred = results["stmt_pred"]

    seg_report = classification_report(
        seg_gt, seg_pred,
        labels=["non_hallucinated", "hallucinated"],
        output_dict=True,
        zero_division=0,
    )

    stmt_acc = accuracy_score(stmt_gt, stmt_pred)

    metrics = {
        "segment_level": {
            "non_hallucinated": seg_report.get("non_hallucinated", {}),
            "hallucinated": seg_report.get("hallucinated", {}),
            "macro_avg": seg_report.get("macro avg", {}),
            "weighted_avg": seg_report.get("weighted avg", {}),
            "total_segments": len(seg_gt),
        },
        "statement_level": {
            "accuracy": stmt_acc,
            "total_statements": len(stmt_gt),
        },
        "error_type_distribution": {},
    }

    # Error type counts for predicted hallucinations
    for et, pred in zip(results["seg_error_types"], seg_pred):
        if pred == "hallucinated":
            metrics["error_type_distribution"][et] = (
                metrics["error_type_distribution"].get(et, 0) + 1
            )

    return metrics


def plot_confusion_matrix(results: dict, save_path: Path):
    labels = ["non_hallucinated", "hallucinated"]
    cm = confusion_matrix(results["seg_gt"], results["seg_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted\nnon-hall.", "Predicted\nhall."],
        yticklabels=["True\nnon-hall.", "True\nhall."],
        ax=ax,
    )
    ax.set_title("Segment-Level Confusion Matrix (with RAG)", fontsize=13)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def plot_ablation(metrics_rag: dict, metrics_no_rag: dict, save_path: Path):
    categories = ["precision\n(hall.)", "recall\n(hall.)", "F1\n(hall.)", "stmt acc."]

    def extract(m: dict) -> list[float]:
        seg_hall = m["segment_level"].get("hallucinated", {})
        return [
            seg_hall.get("precision", 0),
            seg_hall.get("recall", 0),
            seg_hall.get("f1-score", 0),
            m["statement_level"]["accuracy"],
        ]

    rag_vals = extract(metrics_rag)
    no_rag_vals = extract(metrics_no_rag)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rag_vals, width, label="With RAG", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, no_rag_vals, width, label="No RAG", color="#DD8452")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: RAG vs No-RAG Retrieval")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ablation chart -> {save_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading benchmark ...")
    df = load_benchmark(N_SAMPLES)
    print(f"Sampled {len(df)} rows")

    print("Initializing detector ...")
    detector = HallucinationDetector(faiss_dir=INDEX_DIR)

    print("\n=== Run 1: With RAG ===")
    results_rag = run_evaluation(detector, df, use_rag=True)
    metrics_rag = compute_metrics(results_rag)

    print("\n=== Run 2: No RAG (ablation) ===")
    results_no_rag = run_evaluation(detector, df, use_rag=False)
    metrics_no_rag = compute_metrics(results_no_rag)

    # Print summary
    print("\n--- Segment-level (with RAG) ---")
    seg = metrics_rag["segment_level"]
    print(f"  Hallucinated  P={seg['hallucinated'].get('precision',0):.3f}  "
          f"R={seg['hallucinated'].get('recall',0):.3f}  "
          f"F1={seg['hallucinated'].get('f1-score',0):.3f}")
    print(f"  Non-Hall.     P={seg['non_hallucinated'].get('precision',0):.3f}  "
          f"R={seg['non_hallucinated'].get('recall',0):.3f}  "
          f"F1={seg['non_hallucinated'].get('f1-score',0):.3f}")
    print(f"  Statement-level accuracy: {metrics_rag['statement_level']['accuracy']:.3f}")

    # Save JSON
    all_metrics = {"with_rag": metrics_rag, "no_rag": metrics_no_rag}
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    # Plots
    plot_confusion_matrix(results_rag, RESULTS_DIR / "confusion_matrix.png")
    plot_ablation(metrics_rag, metrics_no_rag, RESULTS_DIR / "ablation_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
