import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend

# ---------------- SETTINGS ----------------
FS = 256            # sampling frequency
LOW_CUT = 0.5
HIGH_CUT = 5.0
ORDER = 4

RAW_DIR = Path("Data") / "Raw_data"
# -----------------------------------------


def bandpass_ppg(signal, fs=FS, low=LOW_CUT, high=HIGH_CUT, order=ORDER):
    """0.5–5 Hz Butterworth band-pass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    signal = detrend(signal)
    return filtfilt(b, a, signal)


def list_subjects():
    """Return list of subjects with inf_ppg.csv."""
    subs = []
    for s in os.listdir(RAW_DIR):
        if (RAW_DIR / s / "inf_ppg.csv").is_file():
            subs.append(s)
    return sorted(subs)


def visualize_subject(sub_id, seconds=5):
    """Plot raw and filtered PPG for one subject, one trial."""
    file_path = RAW_DIR / sub_id / "inf_ppg.csv"
    df = pd.read_csv(file_path)

    # Pick the FIRST trial (trial 1 column)
    trial_col = df.columns[0]
    raw = df[trial_col].values.astype(float)

    # Filter the signal
    filtered = bandpass_ppg(raw)

    # Extract a window (few seconds)
    N = int(seconds * FS)
    raw_win = raw[:N]
    filt_win = filtered[:N]
    t = np.arange(N) / FS

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t, raw_win, label="Raw signal", color="red", alpha=0.6)
    plt.plot(t, filt_win, label="Filtered (0.5–5 Hz)", color="green", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Subject {sub_id}: Raw vs Filtered PPG (first {seconds} seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    subjects = list_subjects()

    if len(subjects) < 3:
        print("Not enough subjects found!")
        return

    # pick 3 random subjects for visualization
    random_subjects = random.sample(subjects, 3)

    print("Visualizing subjects:", random_subjects)

    for sid in random_subjects:
        visualize_subject(sid, seconds=5)


if __name__ == "__main__":
    main()
