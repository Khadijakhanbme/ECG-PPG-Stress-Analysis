import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------
# LOAD ECG FEATURES
# -------------------------------------------
DATA_PATH = Path("Data") / "Features" / "ECG_Features_ML.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded ECG dataset:", df.shape)
print(df.head())

# -------------------------------------------
# BASIC CLEANUP
# -------------------------------------------
if "subject" not in df.columns or "label" not in df.columns:
    raise ValueError("Your CSV must contain at least 'subject' and 'label' columns.")

df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Feature columns = everything except ID columns
id_cols = [c for c in ["subject", "segment", "trial_col", "condition", "label"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in id_cols]

df = df.dropna(subset=feature_cols).copy()

print("\nECG feature columns:", feature_cols)
print("Number of features:", len(feature_cols))

# -------------------------------------------
# ROW-WISE STRATIFIED TRAIN/TEST SPLIT (SUBJECTS MIXED)
# -------------------------------------------
X = df[feature_cols]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\n🔀 Row-wise stratified split (subjects mixed)")
print("Train rows:", len(y_train), "| Test rows:", len(y_test))

print("\nClass distribution in TRAIN:")
print(y_train.value_counts())

print("\nClass distribution in TEST:")
print(y_test.value_counts())

# -------------------------------------------
# MODELS
# -------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

USE_SMOTE = True  # try also False

# -------------------------------------------
# TRAIN AND EVALUATE
# -------------------------------------------
print("\n🚀 Training models...\n")
results = {}
best_name, best_pipe, best_acc = None, None, -1

for name, model in models.items():
    print(f"==================== {name} ====================")

    steps = [("scaler", StandardScaler())]
    if USE_SMOTE:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("model", model))

    pipe = Pipeline(steps=steps)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if probs is not None else None

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    print("\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds), "\n")

    results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

    if acc > best_acc:
        best_acc = acc
        best_name = name
        best_pipe = pipe

# -------------------------------------------
# SUMMARY
# -------------------------------------------
print("\n" + "=" * 60)
print("ECG-ONLY MODEL SUMMARY (Row-wise Stratified Split)")
print("=" * 60)
for name, m in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {m['accuracy']:.4f}")
    print(f"  F1-Score: {m['f1']:.4f}")
    if m["auc"] is not None:
        print(f"  ROC-AUC: {m['auc']:.4f}")

print(f"\n🏆 BEST MODEL: {best_name}  |  Accuracy: {best_acc:.4f}")
print("=" * 60)

# -------------------------------------------
# SAVE BEST PIPELINE
# -------------------------------------------
joblib.dump(best_pipe, "best_ecg_pipeline_rowwise.pkl")
print("\n💾 Saved: best_ecg_pipeline_rowwise.pkl")

# -------------------------------------------
# CONFUSION MATRIX PLOT (BEST)
# -------------------------------------------
best_preds = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0-back", "2-back"],
            yticklabels=["0-back", "2-back"])
plt.title(f"Confusion Matrix - ECG Only ({best_name}) [Row-wise]")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n✨ Done.\n")
