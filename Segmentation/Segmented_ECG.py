import os
from pathlib import Path
import pandas as pd

# ------------ SETTINGS ------------
FS = 256              # Hz
SEG_SEC = 10          # segment length (10 seconds)
TOTAL_MIN = 5         # use first 5 minutes
TOTAL_SEC = TOTAL_MIN * 60

SAMPLES_PER_SEG = FS * SEG_SEC        # 256 * 10 = 2560 samples
N_SEGMENTS = TOTAL_SEC // SEG_SEC     # 300 / 10 = 30 segments

FILT_DIR = Path("Data") / "Filtered_ECG"
OUT_DIR = Path("Data") / "Segmented_ECG"
# ----------------------------------


def list_subjects():
    """Return sorted list of subject IDs that have filtered ECG."""
    subs = []
    for name in os.listdir(FILT_DIR):
        subj_dir = FILT_DIR / name
        if subj_dir.is_dir() and (subj_dir / "inf_ecg_filt.csv").is_file():
            subs.append(name)
    subs = sorted(subs)
    print("Found ECG subjects (filtered):", subs)
    return subs


def segment_subject_to_single_file(subject_id: str):
    """
    For one subject:
      - load filtered ECG (inf_ecg_filt.csv)
      - cut into 10-second segments (first 5 minutes)
      - save ALL segments into ONE CSV: <subject>_segments.csv
    """
    in_path = FILT_DIR / subject_id / "inf_ecg_filt.csv"
    df = pd.read_csv(in_path)

    n_samples = len(df)
    print(f"\nSubject {subject_id}: {n_samples} samples, {df.shape[1]} channels")

    max_segments = min(N_SEGMENTS, n_samples // SAMPLES_PER_SEG)
    if max_segments < N_SEGMENTS:
        print(f"  Warning: only {max_segments} full 10-second segments available")

    all_rows = []

    for seg_idx in range(max_segments):
        start = seg_idx * SAMPLES_PER_SEG
        end = start + SAMPLES_PER_SEG

        seg_df = df.iloc[start:end, :].reset_index(drop=True)

        # add metadata columns
        seg_df.insert(0, "sample", range(len(seg_df)))   # sample index within segment
        seg_df.insert(0, "segment", seg_idx)             # segment number (0..)
        seg_df.insert(0, "subject", subject_id)          # subject ID

        all_rows.append(seg_df)

    if not all_rows:
        print(f"  No full segments for subject {subject_id}, skipping.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{subject_id}_segments.csv"
    full_df.to_csv(out_path, index=False)

    print(f"  → saved {max_segments} segments to: {out_path}")


def main():
    subjects = list_subjects()
    if not subjects:
        print("No filtered ECG subjects found in", FILT_DIR)
        return

    for sid in subjects:
        segment_subject_to_single_file(sid)

    print("\nECG segmentation complete (10s windows, 1 file per subject).")


if __name__ == "__main__":
    main()


import matplotlib.pyplot as plt
import numpy as np


def plot_one_segment(
    subject_id="002",
    segment_id=0,
    channel_name=None
):
    """
    Plot ONE 10-second ECG segment for visualization (PPT-ready).
    """

    seg_path = OUT_DIR / f"{subject_id}_segments.csv"
    df = pd.read_csv(seg_path)

    # Filter one segment
    seg_df = df[df["segment"] == segment_id]

    # Auto-pick first ECG channel if not specified
    if channel_name is None:
        channel_name = seg_df.columns[3]  # after subject, segment, sample

    ecg = seg_df[channel_name].values
    t = np.arange(len(ecg)) / FS  # time axis (seconds)

    plt.figure(figsize=(10, 4))
    plt.plot(t, ecg, linewidth=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"ECG Segment (Subject {subject_id}, Segment {segment_id}, 10s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------- DISPLAY ONE SEGMENT --------
plot_one_segment(
    subject_id="002",
    segment_id=0   # try 0–29
)
