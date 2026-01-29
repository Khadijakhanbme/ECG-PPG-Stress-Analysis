import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------- SETTINGS ----------------
FS = 256  # Hz
SEG_DIR = Path("Data") / "Segmented_PPG"
OUT_DIR = Path("Data") / "Features"
OUT_FILE = OUT_DIR / "Features_ML.csv"   # <<< UPDATED NAME
# ------------------------------------------


VALID_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"],
}


def list_segmented_subject_files():
    files = [f for f in os.listdir(SEG_DIR) if f.endswith("_segments.csv")]
    return sorted(files)


def infer_condition(col_name: str) -> str:
    low = col_name.lower()
    for cond, keys in VALID_KEYWORDS.items():
        if any(k in low for k in keys):
            return cond
    return "unknown"


def is_valid_column(col_name: str) -> bool:
    return infer_condition(col_name) != "unknown"


# ------------------------------------------------------
def detect_ppg_peaks(sig: np.ndarray, fs: int = FS):
    min_dist = int(0.35 * fs)

    prc95 = np.percentile(sig, 95)
    prc5 = np.percentile(sig, 5)
    peak_range = prc95 - prc5
    if peak_range <= 0:
        return np.array([], dtype=int)

    prominence = 0.25 * peak_range
    height = np.percentile(sig, 80)

    peaks, _ = find_peaks(
        sig,
        distance=min_dist,
        prominence=prominence,
        height=height
    )
    return peaks
# ------------------------------------------------------


def extract_features_from_signal(sig: np.ndarray, fs: int = FS):
    peaks = detect_ppg_peaks(sig, fs)

    if len(peaks) < 3:
        return None

    ibis = []
    amplitudes = []
    crests = []

    for i in range(1, len(peaks)):
        pk = peaks[i]
        pk_prev = peaks[i - 1]

        ibi_sec = (pk - pk_prev) / fs
        ibis.append(ibi_sec)

        seg = sig[pk_prev:pk]
        if len(seg) == 0:
            continue

        trough_rel = np.argmin(seg)
        trough_idx = pk_prev + trough_rel

        amp = sig[pk] - sig[trough_idx]
        crest = (pk - trough_idx) / fs

        amplitudes.append(amp)
        crests.append(crest)

    ibis = np.array(ibis)
    amplitudes = np.array(amplitudes)
    crests = np.array(crests)

    hr = 60.0 / ibis
    ibi_rmssd = np.sqrt(np.mean(np.diff(ibis)**2)) if len(ibis) > 1 else np.nan

    return {
        "n_beats": len(peaks),
        "ibi_mean": float(np.mean(ibis)),
        "ibi_std": float(np.std(ibis)),
        "ibi_var": float(np.var(ibis)),
        "ibi_rmssd": float(ibi_rmssd),
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "amp_mean": float(np.mean(amplitudes)),
        "amp_std": float(np.std(amplitudes)),
        "crest_mean": float(np.mean(crests)),
        "crest_std": float(np.std(crests)),
    }


def process_subject_file(file_name: str):
    subject_id = file_name.replace("_segments.csv", "")
    df = pd.read_csv(SEG_DIR / file_name)

    rows = []

    for seg_num in sorted(df["segment"].unique()):
        seg_df = df[df["segment"] == seg_num]

        ppg_cols = [
            c for c in seg_df.columns
            if c not in ["subject", "segment", "sample"]
            and is_valid_column(c)
        ]

        for col in ppg_cols:
            sig = seg_df[col].values.astype(float)

            feats = extract_features_from_signal(sig)
            if feats is None:
                continue

            cond = infer_condition(col)
            label = 0 if cond == "0-back" else 1

            row = {"subject": subject_id, "label": label}
            row.update(feats)
            rows.append(row)

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_files = list_segmented_subject_files()
    all_rows = []

    for fname in subject_files:
        print(f"Processing: {fname}")
        rows = process_subject_file(fname)
        all_rows.extend(rows)

    if not all_rows:
        print("No features extracted.")
        return

    df = pd.DataFrame(all_rows)

    # Save final ML-ready dataset
    df.to_csv(OUT_FILE, index=False)

    print("\nSaved ML feature file:", OUT_FILE)
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
