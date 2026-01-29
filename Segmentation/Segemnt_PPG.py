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

FILT_DIR = Path("Data") / "Filtered_PPG"
OUT_DIR = Path("Data") / "Segmented_PPG"
# ----------------------------------


def list_subjects():
    subs = []
    for name in os.listdir(FILT_DIR):
        subj_dir = FILT_DIR / name
        if subj_dir.is_dir() and (subj_dir / "inf_ppg_filt.csv").is_file():
            subs.append(name)
    return sorted(subs)


def segment_subject_to_single_file(subject_id: str):
    """Create ONE single CSV containing all segments for a subject."""
    in_path = FILT_DIR / subject_id / "inf_ppg_filt.csv"
    df = pd.read_csv(in_path)

    n_samples = len(df)
    print(f"\nSubject {subject_id}: {n_samples} samples, {df.shape[1]} trials")

    max_segments = min(N_SEGMENTS, n_samples // SAMPLES_PER_SEG)

    all_rows = []

    for seg_idx in range(max_segments):
        start = seg_idx * SAMPLES_PER_SEG
        end   = start + SAMPLES_PER_SEG

        seg_df = df.iloc[start:end, :].reset_index(drop=True)
        seg_df.insert(0, "sample", range(len(seg_df)))     # sample index within segment
        seg_df.insert(0, "segment", seg_idx)               # segment number
        seg_df.insert(0, "subject", subject_id)            # subject ID

        all_rows.append(seg_df)

    # Combine all segments vertically
    full_df = pd.concat(all_rows, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{subject_id}_segments.csv"
    full_df.to_csv(out_path, index=False)

    print(f"  -> saved all {max_segments} segments to: {out_path}")


def main():
    subjects = list_subjects()
    if not subjects:
        print("No subjects found in filtered folder.")
        return

    for sid in subjects:
        segment_subject_to_single_file(sid)

    print("\nAll subjects segmented and saved to individual files!")


if __name__ == "__main__":
    main()
