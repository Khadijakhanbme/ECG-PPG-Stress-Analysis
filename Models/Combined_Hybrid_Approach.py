import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# ===================== SETTINGS =====================
FS = 256
SEG_DIR_PPG = Path("Data") / "Segmented_PPG"
SEG_DIR_ECG = Path("Data") / "Segmented_ECG"
OUT_DIR = Path("Data") / "Features"
OUT_FILE = OUT_DIR / "PPG_ECG_PAT_combined_features.csv"

VALID_KEYWORDS = {
    "0-back": ["0b", "0back"],
    "2-back": ["2b", "2back"]
}

USE_SMOTE = True
RANDOM_STATE = 42
# ===================================================


# ------------------ HELPERS ------------------
def infer_condition(col_name: str) -> str:
    low = col_name.lower()
    for cond, keys in VALID_KEYWORDS.items():
        if any(k in low for k in keys):
            return cond
    return "unknown"


def is_valid_column(col_name: str) -> bool:
    return infer_condition(col_name) != "unknown"


def list_common_subjects():
    ppg_files = {f.replace("_segments.csv", "") for f in os.listdir(SEG_DIR_PPG) if f.endswith("_segments.csv")}
    ecg_files = {f.replace("_segments.csv", "") for f in os.listdir(SEG_DIR_ECG) if f.endswith("_segments.csv")}
    common = sorted(ppg_files & ecg_files)
    print("Subjects with BOTH PPG & ECG:", common)
    return common


def detect_peaks_robust(sig: np.ndarray, fs: int = FS):
    min_dist = int(0.35 * fs)
    prc95 = np.percentile(sig, 95)
    prc5 = np.percentile(sig, 5)
    peak_range = prc95 - prc5
    if peak_range <= 0:
        return np.array([], dtype=int)

    peaks, _ = find_peaks(
        sig,
        distance=min_dist,
        prominence=0.25 * peak_range,
        height=np.percentile(sig, 80)
    )
    return peaks


# ------------------ PPG FEATURES ------------------
def extract_ppg_features(sig):
    peaks = detect_peaks_robust(sig)
    if len(peaks) < 3:
        return None

    ibis, amps, crests = [], [], []

    for i in range(1, len(peaks)):
        pk, prev = peaks[i], peaks[i - 1]
        ibi = (pk - prev) / FS
        ibis.append(ibi)

        seg = sig[prev:pk]
        if len(seg) == 0:
            continue
        trough = prev + np.argmin(seg)
        amps.append(sig[pk] - sig[trough])
        crests.append((pk - trough) / FS)

    ibis = np.array(ibis)
    hr = 60.0 / ibis

    return {
        "ppg_n_beats": len(peaks),
        "ppg_ibi_mean": np.mean(ibis),
        "ppg_ibi_std": np.std(ibis, ddof=1) if len(ibis) > 1 else 0,
        "ppg_ibi_rmssd": np.sqrt(np.mean(np.diff(ibis) ** 2)) if len(ibis) > 1 else np.nan,
        "ppg_hr_mean": np.mean(hr),
        "ppg_hr_std": np.std(hr, ddof=1) if len(hr) > 1 else 0,
        "ppg_amp_mean": np.mean(amps),
        "ppg_amp_std": np.std(amps, ddof=1) if len(amps) > 1 else 0,
        "ppg_crest_mean": np.mean(crests),
        "ppg_crest_std": np.std(crests, ddof=1) if len(crests) > 1 else 0,
    }


# ------------------ ECG FEATURES ------------------
def extract_ecg_features(r_peaks):
    if len(r_peaks) < 3:
        return None

    rr = np.diff(r_peaks) / FS
    hr = 60.0 / rr

    diff_rr = np.diff(rr) if len(rr) > 1 else []

    return {
        "ecg_n_rpeaks": len(r_peaks),
        "ecg_rr_mean": np.mean(rr),
        "ecg_rr_std": np.std(rr, ddof=1) if len(rr) > 1 else 0,
        "ecg_rr_rmssd": np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else np.nan,
        "ecg_rr_pnn50": np.mean(np.abs(diff_rr) > 0.05) if len(diff_rr) > 0 else np.nan,
        "ecg_hr_mean": np.mean(hr),
        "ecg_hr_std": np.std(hr, ddof=1) if len(hr) > 1 else 0,
    }


# ------------------ PAT ------------------
def extract_pat(r_peaks, ppg_peaks):
    pats = []
    for r in r_peaks:
        after = ppg_peaks[ppg_peaks > r]
        if len(after) > 0:
            pat = (after[0] - r) / FS
            if 0.05 <= pat <= 0.5:
                pats.append(pat)

    return {
        "pat_mean": np.mean(pats) if pats else np.nan,
        "pat_std": np.std(pats, ddof=1) if len(pats) > 1 else 0,
        "n_pat_pairs": len(pats),
    }


# ------------------ BUILD FEATURE TABLE ------------------
def build_feature_table():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = list_common_subjects()
    rows = []

    for sid in subjects:
        df_ppg = pd.read_csv(SEG_DIR_PPG / f"{sid}_segments.csv")
        df_ecg = pd.read_csv(SEG_DIR_ECG / f"{sid}_segments.csv")

        for seg in sorted(set(df_ppg["segment"]) & set(df_ecg["segment"])):
            seg_ppg = df_ppg[df_ppg["segment"] == seg]
            seg_ecg = df_ecg[df_ecg["segment"] == seg]

            cols = [c for c in seg_ppg.columns if is_valid_column(c)]

            for col in cols:
                label = 0 if infer_condition(col) == "0-back" else 1

                ppg_sig = seg_ppg[col].values.astype(float)
                ecg_sig = seg_ecg[col].values.astype(float)

                ppg_feats = extract_ppg_features(ppg_sig)
                r_peaks = detect_peaks_robust(ecg_sig)
                ecg_feats = extract_ecg_features(r_peaks)

                if ppg_feats is None or ecg_feats is None:
                    continue

                pat_feats = extract_pat(r_peaks, detect_peaks_robust(ppg_sig))

                row = {
                    "subject": sid,
                    "segment": seg,
                    "trial_col": col,
                    "label": label
                }
                row.update(ppg_feats)
                row.update(ecg_feats)
                row.update(pat_feats)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Features saved → {OUT_FILE} | shape={df.shape}")
    return df


# ------------------ TRAINING + FEATURE SELECTION ------------------
def train_models(df):
    id_cols = ["subject", "segment", "trial_col", "label"]
    feature_cols = [c for c in df.columns if c not in id_cols]

    # within-subject split
    rng = np.random.default_rng(RANDOM_STATE)
    train, test = [], []

    for sid, g in df.groupby("subject"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        cut = int(0.75 * len(idx))
        train.append(df.loc[idx[:cut]])
        test.append(df.loc[idx[cut:]])

    train_df = pd.concat(train)
    test_df = pd.concat(test)

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    # -------- FEATURE SELECTION --------
    print("\n🔍 Feature Selection (Random Forest importance)")
    fs = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    fs.fit(X_train, y_train)

    selector = SelectFromModel(fs, threshold="median", prefit=True)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    selected_features = np.array(feature_cols)[selector.get_support()]
    print(f"Selected {len(selected_features)} features:")
    print(selected_features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            verbosity=0,
            random_state=RANDOM_STATE
        )
    }

    best_acc, best_pipe = -1, None
    best_name = None
    best_preds = None

    for name, model in models.items():
        print(f"\n==== {name} ====")
        steps = [("scaler", StandardScaler())]
        if USE_SMOTE:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", model))

        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_pipe = pipe
            best_name = name
            best_preds = preds

    # -------- CONFUSION MATRIX FOR BEST MODEL --------
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0-back", "2-back"],
        yticklabels=["0-back", "2-back"]
    )
    plt.title(f"Confusion Matrix — Best Model: {best_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    joblib.dump(best_pipe, "best_combined_pipeline_with_FS.pkl")
    print(f"\n💾 Saved best model | {best_name} | Accuracy={best_acc:.4f}")


# ------------------ MAIN ------------------
def main():
    df = build_feature_table()
    train_models(df)


if __name__ == "__main__":
    main()
