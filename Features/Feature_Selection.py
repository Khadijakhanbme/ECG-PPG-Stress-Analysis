"""
Feature_Selection_Top10.py

Automatically selects the Top-10 most important features using Random Forest:
- Detects target + subject columns
- Subject-wise train/test split
- Scales train only
- Optional SMOTE
- Fits RandomForest
- Prints Top-10 features
- Displays bar chart

No CSV is saved — results are printed only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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
# AUTO DETECT TARGET + SUBJECT
# -------------------------------------------
def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

target_col = detect_column(df, ["label", "Label", "target", "class"])
subject_col = detect_column(df, ["subject", "Subject", "subject_id", "file", "filename"])

if target_col is None or subject_col is None:
    raise ValueError("Could not detect target or subject column automatically.")

print(f"\nDetected target column: {target_col}")
print(f"Detected subject column: {subject_col}")


# -------------------------------------------
# SUBJECT-WISE SPLIT
# -------------------------------------------
subjects = df[subject_col].unique()
np.random.seed(42)
np.random.shuffle(subjects)

train_subjects = subjects[:int(len(subjects)*0.75)]
test_subjects = subjects[int(len(subjects)*0.25):]

train_df = df[df[subject_col].isin(train_subjects)]
test_df = df[df[subject_col].isin(test_subjects)]

X_train = train_df.drop(columns=[target_col, subject_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col, subject_col])
y_test = test_df[target_col]

print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")


# -------------------------------------------
# SCALE TRAIN ONLY
# -------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------
# OPTIONAL SMOTE
# -------------------------------------------
print("\nApplying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print("After SMOTE distribution:")
print(pd.Series(y_train_res).value_counts())


# -------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

importances = rf.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top10 = importance_df.head(10)

print("\n==============================")
print(" TOP 10 MOST IMPORTANT FEATURES")
print("==============================")
print(top10)


# -------------------------------------------
# PLOT BAR CHART
# -------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=top10, x="importance", y="feature", palette="Blues_r")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
