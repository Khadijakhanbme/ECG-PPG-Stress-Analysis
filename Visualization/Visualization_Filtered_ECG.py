import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS = 256  # Hz
RAW_DIR = Path("Data") / "Raw_data"
FILT_DIR = Path("Data") / "Filtered_ECG"

NUM_SUBJECTS_TO_SHOW = 3
WINDOW_SEC = 12     # 10–15 seconds window
# ------------------------------------------


def list_ecg_subjects():
    """Subjects that have BOTH raw and filtered ECG."""
    subjects = []
    for d in os.listdir(RAW_DIR):
        raw_file = RAW_DIR / d / "inf_ecg.csv"
        filt_file = FILT_DIR / d / "inf_ecg_filt.csv"
        if raw_file.is_file() and filt_file.is_file():
            subjects.append(d)
    subjects = sorted(subjects)
    print("Subjects with raw+filtered ECG:", subjects)
    return subjects


def visualize_subject(subject_id: str):
    """Plot raw vs filtered ECG for a random trial and 10–15 s window."""
    raw_path = RAW_DIR / subject_id / "inf_ecg.csv"
    filt_path = FILT_DIR / subject_id / "inf_ecg_filt.csv"

    df_raw = pd.read_csv(raw_path)
    df_filt = pd.read_csv(filt_path)

    # pick a random trial/column
    col = random.choice(df_raw.columns)

    sig_raw = df_raw[col].values.astype(float)
    sig_filt = df_filt[col].values.astype(float)

    # choose window
    win_n = WINDOW_SEC * FS
    if len(sig_raw) > win_n:
        start = random.randint(0, len(sig_raw) - win_n)
    else:
        start = 0
    end = start + win_n

    raw_win = sig_raw[start:end]
    filt_win = sig_filt[start:end]

    # --- plot ---
    plt.figure(figsize=(12, 4))
    plt.plot(raw_win, label="Raw ECG", alpha=0.7)
    plt.plot(filt_win, label="Filtered ECG", linewidth=1.2)

    plt.title(f"ECG Filtering Check (Random 10–15 s Window)\n"
              f"Subject: {subject_id} | Trial: {col}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    subjects = list_ecg_subjects()
    if not subjects:
        print("❌ No subjects with both raw & filtered ECG found.")
        return

    chosen = random.sample(subjects, min(NUM_SUBJECTS_TO_SHOW, len(subjects)))
    print("Visualizing subjects:", chosen)

    for sid in chosen:
        visualize_subject(sid)


if __name__ == "__main__":
    main()
