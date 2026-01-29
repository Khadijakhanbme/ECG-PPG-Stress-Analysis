import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------- SETTINGS ----------------
FS = 256  # Hz

SEG_DIR_PPG = Path("Data") / "Segmented_PPG"
SEG_DIR_ECG = Path("Data") / "Segmented_ECG"
OUT_DIR = Path("Data") / "Features"

OUT_FILE_ECG = OUT_DIR / "ECG_Features_ML.csv"
OUT_FILE_PAT = OUT_DIR / "ECG_PAT_Features_ML.csv"

# Only use 0-back and 2-back
VALID_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"],
}
# ------------------------------------------


def infer_condition(col_name: str) -> str:
    low = col_name.lower()
    for cond, keys in VALID_KEYWORDS.items():
        if any(k in low for k in keys):
            return cond
    return "unknown"


def is_valid_column(col_name: str) -> bool:
    return infer_condition(col_name) != "unknown"


def list_segmented_subjects_both():
    """Subjects that have BOTH PPG & ECG segmented files."""
    ppg_files = {f.replace("_segments.csv", "") for f in os.listdir(SEG_DIR_PPG)
                 if f.endswith("_segments.csv")}
    ecg_files = {f.replace("_segments.csv", "") for f in os.listdir(SEG_DIR_ECG)
                 if f.endswith("_segments.csv")}

    common = sorted(ppg_files & ecg_files)
    print("Subjects with BOTH PPG & ECG:", common)
    return common


# ------------------------------------------------------
#  UNIFIED ROBUST PEAK DETECTION
# ------------------------------------------------------
def detect_peaks_robust(sig: np.ndarray, fs: int = FS):
    """Used for both R-peaks (ECG) and systolic peaks (PPG)."""
    min_dist = int(0.35 * fs)

    prc95 = np.percentile(sig, 95)
    prc5 = np.percentile(sig, 5)
    peak_range = prc95 - prc5
    if peak_range <= 0:
        return np.array([], dtype=int)

    prom = 0.25 * peak_range
    height = np.percentile(sig, 80)

    peaks, _ = find_peaks(
        sig,
        distance=min_dist,
        prominence=prom,
        height=height,
    )
    return peaks


# ------------------------------------------------------
#    ECG FEATURES (NO PAT)
# ------------------------------------------------------
def extract_ecg_hrv_features(r_peaks: np.ndarray, fs: int = FS):
    if len(r_peaks) < 3:
        return None

    rr = np.diff(r_peaks) / fs  # seconds

    if len(rr) == 0:
        return None

    hr = 60.0 / rr  # bpm

    if len(rr) > 1:
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

        nn50 = np.sum(np.abs(diff_rr) > 0.05)   # 50 ms
        pnn50 = nn50 / len(diff_rr)
    else:
        rmssd = np.nan
        pnn50 = np.nan

    return {
        "n_rpeaks": len(r_peaks),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
        "rr_var": float(np.var(rr, ddof=1)) if len(rr) > 1 else 0.0,
        "rr_rmssd": rmssd,
        "rr_pnn50": pnn50,
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr, ddof=1)) if len(hr) > 1 else 0.0,
    }


# ------------------------------------------------------
#    PAT FEATURES (ECG + PPG)
# ------------------------------------------------------
def compute_pat(r_peaks, ppg_peaks, fs=FS):
    pats = []

    for r in r_peaks:
        after = ppg_peaks[ppg_peaks > r]
        if len(after) == 0:
            continue

        p = after[0]
        pat = (p - r) / fs  # seconds

        if 0.05 <= pat <= 0.5:
            pats.append(pat)

    if len(pats) == 0:
        return {"pat_mean": np.nan, "pat_std": np.nan, "n_pat_pairs": 0}

    pats = np.array(pats)
    return {
        "pat_mean": float(np.mean(pats)),
        "pat_std": float(np.std(pats, ddof=1)) if len(pats) > 1 else 0.0,
        "n_pat_pairs": len(pats),
    }


# ------------------------------------------------------
#    PROCESS SUBJECT
# ------------------------------------------------------
def process_subject(subject_id):
    ppg_path = SEG_DIR_PPG / f"{subject_id}_segments.csv"
    ecg_path = SEG_DIR_ECG / f"{subject_id}_segments.csv"

    df_ppg = pd.read_csv(ppg_path)
    df_ecg = pd.read_csv(ecg_path)

    rows_ecg = []
    rows_pat = []

    common_segments = sorted(set(df_ppg["segment"]) & set(df_ecg["segment"]))

    for seg in common_segments:
        seg_ppg = df_ppg[df_ppg["segment"] == seg]
        seg_ecg = df_ecg[df_ecg["segment"] == seg]

        ppg_cols = [c for c in seg_ppg.columns
                    if c not in ["subject", "segment", "sample"] and is_valid_column(c)]
        ecg_cols = [c for c in seg_ecg.columns
                    if c not in ["subject", "segment", "sample"] and is_valid_column(c)]

        common_cols = sorted(set(ppg_cols) & set(ecg_cols))

        for col in common_cols:
            ppg_sig = seg_ppg[col].values.astype(float)
            ecg_sig = seg_ecg[col].values.astype(float)

            r_peaks = detect_peaks_robust(ecg_sig)
            ppg_peaks = detect_peaks_robust(ppg_sig)

            hrv = extract_ecg_hrv_features(r_peaks)
            if hrv is None:
                continue

            condition = infer_condition(col)
            if condition == "unknown":
                continue

            label = 0 if condition == "0-back" else 1

            # ECG-only row — include trial_col + condition like PPG
            row_ecg = {
                "subject": subject_id,
                "segment": int(seg),
                "trial_col": col,
                "condition": condition,
                "label": label,
            }
            row_ecg.update(hrv)
            rows_ecg.append(row_ecg)

            # PAT row — same IDs, plus PAT-specific fields
            pat_feats = compute_pat(r_peaks, ppg_peaks)
            row_pat = dict(row_ecg)
            row_pat.update(pat_feats)
            rows_pat.append(row_pat)

    return rows_ecg, rows_pat


# ------------------------------------------------------
#    MAIN
# ------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = list_segmented_subjects_both()

    all_ecg = []
    all_pat = []

    for sid in subjects:
        print(f"\nProcessing {sid} ...")
        ecg_rows, pat_rows = process_subject(sid)
        all_ecg.extend(ecg_rows)
        all_pat.extend(pat_rows)

    # ----- Save ECG-only -----
    df_ecg = pd.DataFrame(all_ecg)
    if not df_ecg.empty:
        # reorder for clarity
        id_cols = ["subject", "segment", "trial_col", "condition", "label"]
        other_cols = [c for c in df_ecg.columns if c not in id_cols]
        df_ecg = df_ecg[id_cols + other_cols]

    df_ecg.to_csv(OUT_FILE_ECG, index=False)
    print(f"\nSaved ECG-only features → {OUT_FILE_ECG}  shape={df_ecg.shape}")

    # ----- Save ECG+PAT -----
    df_pat = pd.DataFrame(all_pat)
    if not df_pat.empty:
        id_cols = ["subject", "segment", "trial_col", "condition", "label"]
        other_cols = [c for c in df_pat.columns if c not in id_cols]
        df_pat = df_pat[id_cols + other_cols]

    df_pat.to_csv(OUT_FILE_PAT, index=False)
    print(f"Saved ECG+PAT features → {OUT_FILE_PAT}  shape={df_pat.shape}")


if __name__ == "__main__":
    main()
