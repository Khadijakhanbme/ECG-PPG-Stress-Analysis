import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- SETTINGS ----------------
FEAT_DIR = Path("Data") / "Features"

PPG_FILE = FEAT_DIR / "Features_ML.csv"              # PPG-only features
ECG_PAT_FILE = FEAT_DIR / "ECG_PAT_Features_ML.csv"  # ECG + PAT features

OUT_COMBINED = FEAT_DIR / "All_Features_ML.csv"
# ------------------------------------------


def load_and_prefix_ppg(ppg_path: Path):
    """Load PPG features and prefix non-ID columns with 'ppg_'."""
    df = pd.read_csv(ppg_path)

    # ID columns that we DO NOT prefix
    id_cols = [c for c in ["subject", "segment", "trial_col", "condition", "label"]
               if c in df.columns]

    feat_cols = [c for c in df.columns if c not in id_cols]

    df_ppg = df.copy()
    df_ppg.rename(columns={c: f"ppg_{c}" for c in feat_cols}, inplace=True)

    return df_ppg, id_cols


def load_and_prefix_ecg_pat(ecg_pat_path: Path):
    """Load ECG+PAT features and prefix non-ID columns with 'ecg_'."""
    df = pd.read_csv(ecg_pat_path)

    id_cols = [c for c in ["subject", "segment", "trial_col", "condition", "label"]
               if c in df.columns]
    feat_cols = [c for c in df.columns if c not in id_cols]

    df_ecg = df.copy()
    df_ecg.rename(columns={c: f"ecg_{c}" for c in feat_cols}, inplace=True)

    return df_ecg, id_cols


def main():
    if not PPG_FILE.is_file():
        print(f"❌ PPG feature file not found: {PPG_FILE}")
        return
    if not ECG_PAT_FILE.is_file():
        print(f"❌ ECG+PAT feature file not found: {ECG_PAT_FILE}")
        return

    # ---- Load and prefix ----
    df_ppg, id_ppg = load_and_prefix_ppg(PPG_FILE)
    df_ecg, id_ecg = load_and_prefix_ecg_pat(ECG_PAT_FILE)

    # Decide merge keys (must be same trial + segment + subject + label)
    # We try to use the strongest set: subject, segment, trial_col, label
    candidate_keys = ["subject", "segment", "trial_col", "label"]
    merge_keys = [c for c in candidate_keys if c in id_ppg and c in id_ecg]

    if not merge_keys:
        print("❌ No common ID columns for merging (need at least subject/segment/trial_col/label).")
        print("PPG IDs:", id_ppg)
        print("ECG+PAT IDs:", id_ecg)
        return

    print("Merging on keys:", merge_keys)

    df_combined = pd.merge(df_ppg, df_ecg, on=merge_keys, how="inner")

    if df_combined.empty:
        print("❌ Combined feature table is empty after merge. Check IDs/segments/trial_col alignment.")
        return

    # Save combined (with label preserved)
    df_combined.to_csv(OUT_COMBINED, index=False)
    print(f"\n✅ Saved combined feature file → {OUT_COMBINED}")
    print("Shape:", df_combined.shape)
    print(df_combined.head())


if __name__ == "__main__":
    main()
