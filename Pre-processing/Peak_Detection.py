import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# ---------------- SETTINGS ----------------
FS = 256

# PPG segmented directory
SEG_DIR_PPG = Path("Data") / "Segmented_PPG"
# ECG segmented directory
SEG_DIR_ECG = Path("Data") / "Segmented_ECG"

NUM_SUBJECTS_TO_SHOW = 5

# Only use these conditions (NO 3-back) for PPG/ECG columns
VALID_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"]
}
# ------------------------------------------
# ============ COMMON HELPERS ==============

def is_valid_column(col_name: str) -> bool:
    """Return True only for 0-back or 2-back columns."""
    low = col_name.lower()
    for keys in VALID_KEYWORDS.values():
        if any(k in low for k in keys):
            return True
    return False


# ------------------------------------------------------
# ✔ ROBUST PEAK DETECTION (PPG / ECG same logic)
# ------------------------------------------------------
def detect_ppg_peaks(sig, fs=256):
    """
    Robust peak detector used for PPG AND ECG (R-peaks),
    using distance + adaptive prominence + adaptive height.
    """
    # Avoid double peaks — physiological min distance between beats
    min_dist = int(0.35 * fs)

    # Adaptive thresholds (handle different amplitudes across subjects)
    peak_range = np.percentile(sig, 95) - np.percentile(sig, 5)

    if peak_range <= 0:
        return np.array([], dtype=int)

    # Prominence threshold: require a peak to “stand out”
    prom = 0.25 * peak_range  # 25% of dynamic range

    # Height threshold: ignore small noisy bumps
    height = np.percentile(sig, 80)

    peaks, _ = find_peaks(
        sig,
        distance=min_dist,
        prominence=prom,
        height=height
    )
    return peaks
# ------------------------------------------------------


# ============ PPG PART ==============

def list_segmented_ppg_files():
    """Return list of PPG subject segment CSV files."""
    files = [f for f in os.listdir(SEG_DIR_PPG) if f.endswith("_segments.csv")]
    files = sorted(files)
    print("PPG segmented subjects:", files)
    return files


def plot_ppg_with_peaks(subject_id, seg_df, segment_number, col):
    """Plot PPG waveform + detected peaks."""
    sig = seg_df[col].values.astype(float)
    peaks = detect_ppg_peaks(sig, fs=FS)

    print(f"\n🔹 [PPG] Subject: {subject_id}")
    print(f"   Segment: {segment_number}")
    print(f"   Column:  {col}")
    print(f"   Peaks detected: {len(peaks)}")

    plt.figure(figsize=(12, 4))
    plt.plot(sig, label="PPG Signal", linewidth=1)
    if len(peaks) > 0:
        plt.plot(peaks, sig[peaks], "ro", label="Detected Peaks")

    plt.title(f"PPG Peak Detection (0-back / 2-back Only)\n"
              f"Subject: {subject_id} | Segment: {segment_number} | Column: {col}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_ppg():
    subject_files = list_segmented_ppg_files()
    if len(subject_files) == 0:
        print("❌ No segmented PPG subject files found.")
        return

    subjects_to_show = random.sample(subject_files, min(NUM_SUBJECTS_TO_SHOW, len(subject_files)))

    for file in subjects_to_show:
        subject_id = file.replace("_segments.csv", "")
        path = SEG_DIR_PPG / file

        df = pd.read_csv(path)

        # Choose random available segment
        segments_list = df["segment"].unique()
        seg_num = random.choice(segments_list)

        seg_df = df[df["segment"] == seg_num]

        # Select only 0-back and 2-back trial columns
        ppg_cols = [
            c for c in seg_df.columns
            if c not in ["subject", "segment", "sample"] and is_valid_column(c)
        ]

        if len(ppg_cols) == 0:
            print(f"⚠ No valid PPG 0/2-back columns for subject {subject_id}")
            continue

        col = random.choice(ppg_cols)

        plot_ppg_with_peaks(subject_id, seg_df, seg_num, col)


# ============ ECG PART ==============

def list_segmented_ecg_files():
    """Return list of ECG subject segment CSV files."""
    files = [f for f in os.listdir(SEG_DIR_ECG) if f.endswith("_segments.csv")]
    files = sorted(files)
    print("ECG segmented subjects:", files)
    return files


def plot_ecg_with_peaks(subject_id, seg_df, segment_number, col):
    """Plot ECG waveform + detected R-peaks."""
    sig = seg_df[col].values.astype(float)
    # Use SAME logic as PPG
    peaks = detect_ppg_peaks(sig, fs=FS)

    print(f"\n🔹 [ECG] Subject: {subject_id}")
    print(f"   Segment: {segment_number}")
    print(f"   Column:  {col}")
    print(f"   R-peaks detected: {len(peaks)}")

    plt.figure(figsize=(12, 4))
    plt.plot(sig, label="ECG Signal", linewidth=1)
    if len(peaks) > 0:
        plt.plot(peaks, sig[peaks], "ro", label="R-peaks")

    plt.title(f"ECG R-peak Detection\n"
              f"Subject: {subject_id} | Segment: {segment_number} | Column: {col}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_ecg():
    subject_files = list_segmented_ecg_files()
    if len(subject_files) == 0:
        print("❌ No segmented ECG subject files found.")
        return

    subjects_to_show = random.sample(subject_files, min(NUM_SUBJECTS_TO_SHOW, len(subject_files)))

    for file in subjects_to_show:
        subject_id = file.replace("_segments.csv", "")
        path = SEG_DIR_ECG / file

        df = pd.read_csv(path)

        # Choose random available segment
        segments_list = df["segment"].unique()
        seg_num = random.choice(segments_list)

        seg_df = df[df["segment"] == seg_num]

        # ECG: take all signal columns or (if you want) only 0/2-back like PPG
        ecg_cols = [
            c for c in seg_df.columns
            if c not in ["subject", "segment", "sample"]
            # Uncomment next line if ECG columns also have 0b/2b naming and
            # you want to restrict to those:
            # and is_valid_column(c)
        ]

        if len(ecg_cols) == 0:
            print(f"⚠ No ECG columns for subject {subject_id}")
            continue

        col = random.choice(ecg_cols)

        plot_ecg_with_peaks(subject_id, seg_df, seg_num, col)


# ============ MAIN ==============

def main():
    print("\n=== PPG Peak Detection & Visualization ===")
    visualize_ppg()

    print("\n=== ECG R-peak Detection & Visualization ===")
    visualize_ecg()


if __name__ == "__main__":
    main()
