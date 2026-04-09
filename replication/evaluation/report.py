"""
Scoring and reporting for IMO grading.

Reimplemented from facebookresearch/HyperAgents domains/report.py.
Same accuracy computation, same MAE with point mapping, same per-label stats.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

QUESTION_ID = "Grading ID"
GROUND_TRUTH_KEY = "Reward"

# Paper's point mapping for MAE
REWARD_TO_POINTS = {
    "incorrect": 0,
    "partial": 1,
    "almost": 6,
    "correct": 7,
}
MAX_ERROR = 7


def compute_report(predictions_path: str) -> dict:
    """Compute accuracy and MAE from predictions CSV.

    Matches paper's domains/report.py exactly for imo_grading domain.

    Returns:
        Report dict with overall_accuracy, normalized_mae, per-label stats.
    """
    df = pd.read_csv(predictions_path, dtype=str)
    df = df[df["prediction"] != ""].copy()
    df["prediction"] = df["prediction"].str.strip().str.lower()
    df[GROUND_TRUTH_KEY] = df[GROUND_TRUTH_KEY].str.strip().str.lower()
    df["match"] = df[GROUND_TRUTH_KEY] == df["prediction"]

    accuracy = float(df["match"].mean())
    total_correct = int(df["match"].sum())
    total = len(df)

    logger.info("Accuracy: %.3f (%d/%d)", accuracy, total_correct, total)

    # MAE with point mapping
    df["pred_points"] = df["prediction"].map(REWARD_TO_POINTS).astype(float)
    df["true_points"] = df[GROUND_TRUTH_KEY].map(REWARD_TO_POINTS).astype(float)
    df["error"] = abs(df["pred_points"] - df["true_points"])
    df["error"] = df["error"].fillna(MAX_ERROR)
    mae = float(df["error"].mean() / MAX_ERROR)

    logger.info("Normalized MAE: %.3f", mae)

    # Per-label precision/recall
    labels = sorted(set(df[GROUND_TRUTH_KEY].unique()))
    label_report = {}
    for label in labels:
        tp = int(((df["prediction"] == label) & (df[GROUND_TRUTH_KEY] == label)).sum())
        fp = int(((df["prediction"] == label) & (df[GROUND_TRUTH_KEY] != label)).sum())
        fn = int(((df["prediction"] != label) & (df[GROUND_TRUTH_KEY] == label)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        total_label = int((df[GROUND_TRUTH_KEY] == label).sum())
        label_report[label] = {
            "precision": precision,
            "recall": recall,
            "correct": tp,
            "total": total_label,
        }

    report = {
        "overall_accuracy": accuracy,
        "normalized_mean_absolute_error": mae,
        "total_correct": total_correct,
        "total": total,
        "accuracy_by_label": label_report,
    }

    # Save report
    report_path = predictions_path.replace("predictions.csv", "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
