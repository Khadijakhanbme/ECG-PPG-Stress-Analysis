import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
DATA_PATH = Path("Data") / "Features" / "All_Features_ML.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded dataset:", df.shape)
print(df.head())

# -------------------------------------------
# SELECT TOP 10 FEATURES
# -------------------------------------------
top_10_features = [
    "ppg_amp_mean",
    "ecg_pat_mean",
    "ppg_crest_mean",
    "ppg_amp_std",
    "ppg_hr_mean",
    "ppg_ibi_mean",
    "ecg_rr_mean",
    "ecg_hr_mean",
    "ppg_ibi_rmssd",
    "ecg_rr_rmssd"
]

print("\nUsing Top 10 Features:")
print(top_10_features)

# -------------------------------------------
# SUBJECT-WISE TRAIN/TEST SPLIT (NO LEAKAGE)
# -------------------------------------------
subjects = df["subject"].unique()
rng = np.random.default_rng(42)
rng.shuffle(subjects)

cut = int(len(subjects) * 0.75)
train_subjects = subjects[:cut]
test_subjects = subjects[cut:]  # FIXED: no overlap

# sanity check: no subject appears in both sets
assert set(train_subjects).isdisjoint(set(test_subjects)), "Leakage: train/test subjects overlap!"

train_df = df[df["subject"].isin(train_subjects)].copy()
test_df = df[df["subject"].isin(test_subjects)].copy()

X_train = train_df[top_10_features]
y_train = train_df["label"]

X_test = test_df[top_10_features]
y_test = test_df["label"]

print(f"\n🔀 Splitting dataset by subject...")
print(f"Train subjects: {len(train_subjects)}, Test subjects: {len(test_subjects)}")
print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")

print("\nClass distribution in TRAIN (before SMOTE):")
print(y_train.value_counts())

print("\nClass distribution in TEST:")
print(y_test.value_counts())

# -------------------------------------------
# DEFINE MODELS
# NOTE: Since we use SMOTE, don't also use class_weight/scale_pos_weight (avoid double balancing)
# -------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
}

# -------------------------------------------
# TRAIN + EVALUATE
# Pipeline order: Scaling -> SMOTE (train only) -> Model
# -------------------------------------------
print("\n🚀 Training models...\n")
results = {}
best_pipeline = None
best_accuracy = -1
best_model_name = ""

for name, model in models.items():
    print(f"==================== {name} ====================")

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    # AUC needs probabilities
    probs = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_test)[:, 1]

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

    if acc > best_accuracy:
        best_accuracy = acc
        best_pipeline = pipe
        best_model_name = name

# -------------------------------------------
# DISPLAY RESULTS SUMMARY
# -------------------------------------------
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    if metrics["auc"] is not None:
        print(f"  ROC-AUC: {metrics['auc']:.4f}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"➡ Best Accuracy: {best_accuracy:.4f}")
print("=" * 60)

# -------------------------------------------
# SAVE BEST PIPELINE (Scaler + SMOTE + Model)
# -------------------------------------------
joblib.dump(best_pipeline, "best_stress_pipeline.pkl")
print("\n💾 Best pipeline saved as: best_stress_pipeline.pkl")

# -------------------------------------------
# CONFUSION MATRIX FOR BEST MODEL
# -------------------------------------------
best_preds = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0-back", "2-back"],
            yticklabels=["0-back", "2-back"])
plt.title(f"Confusion Matrix - {best_model_name} (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n✨ Done.\n")
