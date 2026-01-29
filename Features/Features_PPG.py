import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------- SETTINGS ----------------
FS = 256  # Hz
SEG_DIR = Path("Data") / "Segmented_PPG"
OUT_DIR = Path("Data") / "Features"
OUT_FILE = OUT_DIR / "ppg_10s_features_0vs2.csv"

# Only use these conditions (NO 3-back)
VALID_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"],
}
# ------------------------------------------


def list_segmented_subject_files():
    """Return list of subject segment CSV files."""
    files = [f for f in os.listdir(SEG_DIR) if f.endswith("_segments.csv")]
    files = sorted(files)
    print("Segmented subjects:", files)
    return files


def infer_condition(col_name: str) -> str:
    """Map column name to '0-back' or '2-back' (or 'unknown')."""
    low = col_name.lower()
    for cond, keys in VALID_KEYWORDS.items():
        if any(k in low for k in keys):
            return cond
    return "unknown"


def is_valid_column(col_name: str) -> bool:
    """Return True only for 0-back or 2-back columns."""
    return infer_condition(col_name) != "unknown"


# ------------------------------------------------------
#  ROBUST PPG PEAK DETECTION (distance + prominence + height)
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
        height=height,
    )
    return peaks
# ------------------------------------------------------


def extract_features_from_signal(sig: np.ndarray, fs: int = FS):
    peaks = detect_ppg_peaks(sig, fs=fs)

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
        crest_time_sec = (pk - trough_idx) / fs

        amplitudes.append(amp)
        crests.append(crest_time_sec)

    ibis = np.array(ibis)
    amplitudes = np.array(amplitudes)
    crests = np.array(crests)

    hr = 60.0 / ibis

    if len(ibis) >= 2:
        diff_ibi = np.diff(ibis)
        ibi_rmssd = float(np.sqrt(np.mean(diff_ibi ** 2)))
    else:
        ibi_rmssd = np.nan

    feats = {
        "n_beats": int(len(peaks)),
        "segment_duration_sec": len(sig) / fs,

        "ibi_mean": float(np.mean(ibis)),
        "ibi_std": float(np.std(ibis, ddof=1)) if len(ibis) > 1 else 0.0,
        "ibi_var": float(np.var(ibis, ddof=1)) if len(ibis) > 1 else 0.0,
        "ibi_rmssd": ibi_rmssd,

        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr, ddof=1)) if len(hr) > 1 else 0.0,

        "amp_mean": float(np.mean(amplitudes)),
        "amp_std": float(np.std(amplitudes, ddof=1)) if len(amplitudes) > 1 else 0.0,

        "crest_mean": float(np.mean(crests)),
        "crest_std": float(np.std(crests, ddof=1)) if len(crests) > 1 else 0.0,
    }

    return feats


def process_subject_file(file_name: str):
    subject_id = file_name.replace("_segments.csv", "")
    path = SEG_DIR / file_name

    df = pd.read_csv(path)
    rows = []

    segment_ids = sorted(df["segment"].unique())

    for seg_num in segment_ids:
        seg_df = df[df["segment"] == seg_num]

        ppg_cols = [
            c for c in seg_df.columns
            if c not in ["subject", "segment", "sample"] and is_valid_column(c)
        ]
        if not ppg_cols:
            continue

        for col in ppg_cols:
            sig = seg_df[col].values.astype(float)
            feats = extract_features_from_signal(sig, fs=FS)
            if feats is None:
                continue

            condition = infer_condition(col)
            if condition == "unknown":
                continue

            # LABEL: 0 for 0-back, 1 for 2-back
            label = 0 if condition == "0-back" else 1

            # ⚠ These ID columns MUST stay in the file for safe merge:
            #   subject, segment, trial_col, label (and condition if you like)
            row = {
                "subject": subject_id,
                "segment": int(seg_num),
                "trial_col": col,
                "condition": condition,
                "label": label,
            }
            row.update(feats)
            rows.append(row)

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_files = list_segmented_subject_files()
    if not subject_files:
        print("No segmented subject files found.")
        return

    all_rows = []

    for file_name in subject_files:
        print(f"\nProcessing subject file: {file_name}")
        rows = process_subject_file(file_name)
        all_rows.extend(rows)

    if not all_rows:
        print("No features extracted.")
        return

    features_df = pd.DataFrame(all_rows)

    # Just to be extra sure: order columns with IDs first
    id_cols = ["subject", "segment", "trial_col", "condition", "label"]
    other_cols = [c for c in features_df.columns if c not in id_cols]
    features_df = features_df[id_cols + other_cols]

    features_df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved PPG features to: {OUT_FILE}")
    print("Feature table shape:", features_df.shape)
    print(features_df.head())


if __name__ == "__main__":
    main()
