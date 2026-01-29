import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, detrend
import matplotlib.pyplot as plt

# ----------------- SETTINGS -----------------
FS = 256  # Hz
RAW_DIR = Path("Data") / "Raw_data"

# None = use all subject folders that contain inf_ppg.csv
SUBJECT_IDS = None

# Frequency band where you expect heart-rate peak (for stats)
HR_BAND = (0.5, 3.5)  # Hz
# ---------------------------------------------

# We include all 3 conditions for exploration:
CONDITION_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"],
}


def find_subjects():
    if SUBJECT_IDS is not None:
        return SUBJECT_IDS

    subs = []
    for name in os.listdir(RAW_DIR):
        subj_dir = RAW_DIR / name
        if subj_dir.is_dir() and (subj_dir / "inf_ppg.csv").is_file():
            subs.append(name)
    subs = sorted(subs)
    print("Found subjects:", subs)
    return subs


def assign_columns_to_conditions(df: pd.DataFrame):
    """
    Returns a dict: {condition: [column_names]} based on CONDITION_KEYWORDS.
    Now includes 0-back and 2-back
    """
    cond_cols = {c: [] for c in CONDITION_KEYWORDS.keys()}
    for col in df.columns:
        col_low = col.lower()
        for cond, keys in CONDITION_KEYWORDS.items():
            if any(k in col_low for k in keys):
                cond_cols[cond].append(col)
                break
    return cond_cols


def compute_psd(signal: np.ndarray, fs: int = FS):
    sig = detrend(signal)
    f, Pxx = welch(sig, fs=fs, nperseg=2048)
    return f, Pxx


def main():
    subjects = find_subjects()
    if len(subjects) == 0:
        print("No subjects found in", RAW_DIR)
        return

    # Collect PSDs and HR peaks for each condition
    cond_psd_list = {c: [] for c in CONDITION_KEYWORDS.keys()}
    cond_hr_peaks = {c: [] for c in CONDITION_KEYWORDS.keys()}
    freqs_ref = None

    for sid in subjects:
        path = RAW_DIR / sid / "inf_ppg.csv"
        print(f"\nSubject {sid} → {path}")
        df = pd.read_csv(path)

        cond_cols = assign_columns_to_conditions(df)
        print("  Columns per condition:", cond_cols)

        for cond, cols in cond_cols.items():
            for col in cols:
                sig = df[col].values

                f, Pxx = compute_psd(sig, fs=FS)
                if freqs_ref is None:
                    freqs_ref = f

                cond_psd_list[cond].append(Pxx)

                # find peak frequency in HR_BAND
                band_mask = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
                f_band = f[band_mask]
                P_band = Pxx[band_mask]

                if len(f_band) > 0:
                    f_peak = f_band[np.argmax(P_band)]
                    cond_hr_peaks[cond].append(f_peak)

    # ---- Plotting and statistics ----
    plt.figure(figsize=(10, 4))

    for cond, psds in cond_psd_list.items():
        if len(psds) == 0:
            print(f"\nCondition {cond}: no signals found.")
            continue

        psds = np.vstack(psds)
        mean_psd = psds.mean(axis=0)

        plt.semilogy(freqs_ref, mean_psd, label=f"{cond} (n={psds.shape[0]})")

        # HR STATS
        hr_peaks = np.array(cond_hr_peaks[cond])
        if len(hr_peaks) > 0:
            hr_bpm = hr_peaks * 60
            print(f"\n===== {cond} =====")
            print(f"Signals: {len(hr_peaks)}")
            print(f"Mean HR peak: {hr_peaks.mean():.2f} Hz ({hr_bpm.mean():.1f} bpm)")
            print(f"Std: {hr_peaks.std():.2f} Hz")
            print(f"Min: {hr_peaks.min():.2f} Hz ({hr_bpm.min():.1f} bpm)")
            print(f"Max: {hr_peaks.max():.2f} Hz ({hr_bpm.max():.1f} bpm)")

    plt.xlim(0, 20)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density (log scale)")
    plt.title("PPG Mean Spectrum for 0-back and 2-back condititons")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
