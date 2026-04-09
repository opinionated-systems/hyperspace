"""
Prepare IMO-GradingBench data exactly as the paper does.

Source: domains/imo/curate_subsets.py in facebookresearch/HyperAgents

Steps:
1. Load gradingbench.csv from google-deepmind/superhuman
2. Filter to Points in [0, 1, 6, 7] (clear grades only)
3. Shuffle with random_state=42
4. Take 300 samples, stratified split into 100 train / 100 val / 100 test
5. Save as CSV files

Usage:
    python -m replication.data.prepare
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

GROUND_TRUTH_KEY = "Reward"
DATA_DIR = Path(__file__).parent
SOURCE_CSV = Path(__file__).parent.parent.parent / "datasets" / "imo_grading" / "gradingbench.csv"


def prepare():
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"gradingbench.csv not found at {SOURCE_CSV}. "
            "Run: ./datasets/download.sh"
        )

    df = pd.read_csv(SOURCE_CSV)
    print(f"Original: {len(df)} rows")
    print(f"Points distribution: {df['Points'].value_counts().sort_index().to_dict()}")
    print(f"Reward distribution: {df[GROUND_TRUTH_KEY].value_counts().to_dict()}")

    # Filter: keep only clear grades (Points 0, 1, 6, 7)
    # This matches the paper's curate_subsets.py exactly
    df_filtered = df[df["Points"].isin([0, 1, 6, 7])].reset_index(drop=True)
    df_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFiltered (Points 0,1,6,7): {len(df_shuffled)} rows")
    print(f"Reward distribution: {df_shuffled[GROUND_TRUTH_KEY].value_counts().to_dict()}")

    # Save filtered+shuffled
    df_shuffled.to_csv(DATA_DIR / "gradingbench_filtered.csv", index=False)

    # Stratified split: 100 train / 100 val / 100 test
    subset_size = 100
    n_total = subset_size * 3

    subset = df_shuffled.sample(n=n_total, random_state=42).reset_index(drop=True)

    train_df, temp = train_test_split(
        subset,
        test_size=2 * subset_size,
        random_state=42,
        stratify=subset[GROUND_TRUTH_KEY],
    )
    val_df, test_df = train_test_split(
        temp,
        test_size=subset_size,
        random_state=42,
        stratify=temp[GROUND_TRUTH_KEY],
    )

    print(f"\nTrain: {len(train_df)} rows")
    print(f"  {train_df[GROUND_TRUTH_KEY].value_counts().to_dict()}")
    print(f"Val: {len(val_df)} rows")
    print(f"  {val_df[GROUND_TRUTH_KEY].value_counts().to_dict()}")
    print(f"Test: {len(test_df)} rows")
    print(f"  {test_df[GROUND_TRUTH_KEY].value_counts().to_dict()}")

    # Save splits
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"\nSaved to {DATA_DIR}/")


if __name__ == "__main__":
    prepare()
