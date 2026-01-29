import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend

# ---------------- SETTINGS ----------------
FS = 256            # Sampling frequency (Hz)
LOW_CUT = 0.5       # Band-pass low cutoff (Hz)
HIGH_CUT = 5.0      # Band-pass high cutoff (Hz)
ORDER = 4           # Butterworth filter order

RAW_DIR = Path("Data") / "Raw_data"
OUT_DIR = Path("Data") / "Filtered_PPG"
# -----------------------------------------


def list_subjects():
    """Return sorted list of subject IDs that have inf_ppg.csv."""
    subjects = []
    for name in os.listdir(RAW_DIR):
        subj_dir = RAW_DIR / name
        if subj_dir.is_dir() and (subj_dir / "inf_ppg.csv").is_file():
            subjects.append(name)
    subjects = sorted(subjects)
    print("Found subjects:", subjects)
    return subjects


def bandpass_ppg(signal, fs=FS, low=LOW_CUT, high=HIGH_CUT, order=ORDER):
    """
    Band-pass filter for PPG: 0.5–5 Hz.
    Uses zero-phase filtfilt to avoid phase distortion.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    # remove linear trend before filtering
    signal = detrend(signal)
    filtered = filtfilt(b, a, signal)
    return filtered


def process_subject(subject_id: str):
    """Load one subject's inf_ppg.csv, filter all trials, save new file."""
    in_path = RAW_DIR / subject_id / "inf_ppg.csv"
    df = pd.read_csv(in_path)

    print(f"Subject {subject_id}: filtering {df.shape[1]} trials, {df.shape[0]} samples each")

    df_filt = pd.DataFrame(index=df.index)

    for col in df.columns:
        sig = df[col].values.astype(float)
        sig_filt = bandpass_ppg(sig)
        df_filt[col] = sig_filt

    # make output folder for this subject
    subj_out_dir = OUT_DIR / subject_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = subj_out_dir / "inf_ppg_filt.csv"
    df_filt.to_csv(out_path, index=False)

    print(f"  -> saved filtered PPG to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects()
    if not subjects:
        print("No subjects with inf_ppg.csv found in", RAW_DIR)
        return

    for sid in subjects:
        process_subject(sid)


if __name__ == "__main__":
    main()
