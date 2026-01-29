import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend

# ---------------- SETTINGS ----------------
FS = 256
RAW_DIR = Path("Data") / "Raw_data"
OUT_DIR = Path("Data") / "Filtered_ECG"

LOW_CUT = 0.5      # Hz
HIGH_CUT = 40.0    # Hz
ORDER = 4
# ------------------------------------------


def bandpass_ecg(sig, fs=FS, low=LOW_CUT, high=HIGH_CUT, order=ORDER):
    """Band-pass filter ECG (0.5–40 Hz)."""
    sig = detrend(sig)
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


def process_subject(subject_id: str):
    in_path = RAW_DIR / subject_id / "inf_ecg.csv"
    df = pd.read_csv(in_path)

    print(f"\nSubject {subject_id}: filtering {df.shape[1]} channels, {df.shape[0]} samples")

    df_filt = pd.DataFrame(index=df.index)

    for col in df.columns:
        raw = df[col].values.astype(float)
        filt = bandpass_ecg(raw)
        df_filt[col] = filt

    subj_out = OUT_DIR / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    out_path = subj_out / "inf_ecg_filt.csv"
    df_filt.to_csv(out_path, index=False)
    print(f"  → saved filtered ECG to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = [
        d for d in os.listdir(RAW_DIR)
        if (RAW_DIR / d / "inf_ecg.csv").is_file()
    ]

    print("Found ECG subjects:", subjects)

    for sid in subjects:
        process_subject(sid)


if __name__ == "__main__":
    main()
