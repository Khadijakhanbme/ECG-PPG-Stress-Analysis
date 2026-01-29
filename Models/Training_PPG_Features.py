import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------
# LOAD PPG FEATURES
# -------------------------------------------
DATA_PATH = Path("Data") / "Features" / "ppg_10s_features_0vs2.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded PPG dataset:", df.shape)
print(df.head())

# -------------------------------------------
# BASIC CLEANUP
# -------------------------------------------
if "subject" not in df.columns or "label" not in df.columns:
    raise ValueError("Your CSV must contain at least 'subject' and 'label' columns.")

df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

id_cols = [c for c in ["subject", "segment", "trial_col", "condition", "label"] if c in df.columns]
feature_cols = [c for c in df.columns if c not in id_cols]

df = df.dropna(subset=feature_cols).copy()

print("\nPPG feature columns:", feature_cols)
print("Number of features:", len(feature_cols))

# -------------------------------------------
# WITHIN-SUBJECT SPLIT (EACH SUBJECT IN TRAIN + TEST)
# -------------------------------------------
rng = np.random.default_rng(42)

train_parts = []
test_parts = []

for sid, g in df.groupby(";" \
"454subject"):
    idx = g.index.to_numpy()
    rng.shuffle(idx)

    cut = int(len(idx) * 0.75)  # 75% train, 25% test per subject

    # if subject has too few samples, keep all in train (avoid empty test for that subject)
    if cut < 1 or cut >= len(idx):
        train_parts.append(df.loc[idx])
        continue

    train_parts.append(df.loc[idx[:cut]])
    test_parts.append(df.loc[idx[cut:]])

train_df = pd.concat(train_parts).sample(frac=1, random_state=42)
test_df  = pd.concat(test_parts).sample(frac=1, random_state=42)

X_train, y_train = train_df[feature_cols], train_df["label"]
X_test,  y_test  = test_df[feature_cols],  test_df["label"]

print("\n🔀 Within-subject split (each subject contributes to both sets)")
print("Train rows:", len(y_train), "| Test rows:", len(y_test))

print("\nClass distribution in TRAIN:")
print(y_train.value_counts())

print("\nClass distribution in TEST:")
print(y_test.value_counts())

# Optional: show how many subjects appear in both
print("\nSubjects in train:", train_df["subject"].nunique(),
      "| Subjects in test:", test_df["subject"].nunique(),
      "| Overlap subjects:", len(set(train_df["subject"]) & set(test_df["subject"])) )

# -------------------------------------------
# MODELS
# -------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
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
print("PPG-ONLY MODEL SUMMARY (Within-subject Split)")
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
joblib.dump(best_pipe, "best_ppg_pipeline_withinsubject.pkl")
print("\n💾 Saved: best_ppg_pipeline_withinsubject.pkl")

# -------------------------------------------
# CONFUSION MATRIX PLOT (BEST)
# -------------------------------------------
best_preds = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0-back", "2-back"],
            yticklabels=["0-back", "2-back"])
plt.title(f"Confusion Matrix - PPG Only ({best_name}) [Within-subject]")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n✨ Done.\n")
