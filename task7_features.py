import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Load & confirm target
# ─────────────────────────────────────────
df = pd.read_csv("Placement_Data_Full_Class.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nTarget distribution:")
print(df['status'].value_counts())
print("\nNulls:\n", df.isnull().sum())

# ─────────────────────────────────────────
# STEP 2: Encode & derive features
# ─────────────────────────────────────────
df_feat = df.copy()

# Drop leaky column (salary only exists if placed)
df_feat.drop(columns=['sl_no', 'salary'], inplace=True, errors='ignore')

# Encode binary categoricals
binary_map = {
    'gender':  {'M': 1, 'F': 0},
    'ssc_b':   {'Central': 1, 'Others': 0},
    'hsc_b':   {'Central': 1, 'Others': 0},
    'workex':  {'Yes': 1, 'No': 0},
}
for col, mapping in binary_map.items():
    if col in df_feat.columns:
        df_feat[col] = df_feat[col].map(mapping)

# One-hot encode multi-class categoricals
df_feat = pd.get_dummies(df_feat, columns=['hsc_s', 'degree_t', 'specialisation'], drop_first=True)

# Encode target
df_feat['status'] = (df_feat['status'] == 'Placed').astype(int)

# ── Derived Features ──
# 1. Academic trend: improving from 10th to 12th?
df_feat['score_trend_ssc_hsc'] = df_feat['hsc_p'] - df_feat['ssc_p']

# 2. Overall academic average
df_feat['avg_academic_score'] = df_feat[['ssc_p', 'hsc_p', 'degree_p']].mean(axis=1)

# 3. All scores above 60 flag
df_feat['all_above_60'] = (
    (df_feat['ssc_p'] > 60) &
    (df_feat['hsc_p'] > 60) &
    (df_feat['degree_p'] > 60)
).astype(int)

# 4. Strong MBA + work experience combo
df_feat['mba_with_workex'] = (
    (df_feat['mba_p'] > df_feat['mba_p'].median()) &
    (df_feat['workex'] == 1)
).astype(int)

print("\nTotal features after engineering:", len(df_feat.columns) - 1)

# ─────────────────────────────────────────
# STEP 3: Train / Val / Test Split
# ─────────────────────────────────────────
TARGET = 'status'
X = df_feat.drop(columns=[TARGET])
y = df_feat[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"\nTrain: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────
# STEP 4: Train & Feature Importance
# ─────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\nFeature Importances:")
print(importance_df.to_string())

plt.figure(figsize=(10, 7))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance — Random Forest (Task 7)')
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: results/feature_importance.png")

# ─────────────────────────────────────────
# STEP 5: Leakage Check
# ─────────────────────────────────────────
top_feat       = importance_df.iloc[0]['feature']
top_importance = importance_df.iloc[0]['importance']
train_acc      = rf.score(X_train, y_train)
val_acc        = rf.score(X_val,   y_val)

print(f"\n── Leakage Check ──")
print(f"Top feature : {top_feat}  (importance={top_importance:.3f})")
print(f"Train acc   : {train_acc:.3f}")
print(f"Val acc     : {val_acc:.3f}")
print(f"Gap         : {train_acc - val_acc:.3f}")

if top_importance > 0.4:
    print("⚠️  WARNING: One feature dominates — check for leakage!")
else:
    print("✅  No leakage signal from single feature.")

if train_acc - val_acc > 0.10:
    print("⚠️  WARNING: Train/Val gap > 10% — possible overfit.")
else:
    print("✅  Train/Val gap acceptable.")

# ─────────────────────────────────────────
# STEP 6: Prune & Lock Baseline Feature Set
# ─────────────────────────────────────────
THRESHOLD = 0.02
selected = importance_df[importance_df['importance'] > THRESHOLD]['feature'].tolist()

print(f"\nSelected features ({len(selected)}): {selected}")

with open("results/baseline_features.json", "w") as f:
    json.dump(selected, f, indent=2)
print("✅ Saved: results/baseline_features.json")

# ─────────────────────────────────────────
# STEP 7: Final Report on Selected Features
# ─────────────────────────────────────────
rf_final = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_final.fit(X_train[selected], y_train)

print("\n── Validation Report (selected features only) ──")
print(classification_report(y_val, rf_final.predict(X_val[selected]),
                             target_names=["Not Placed", "Placed"]))

print("\n========== TASK 7 COMPLETE ==========")
print(f"Features engineered : {X.shape[1]}")
print(f"Features selected   : {len(selected)}")
print(f"Val accuracy        : {rf_final.score(X_val[selected], y_val):.3f}")
print("Outputs in          : results/")
print("=====================================")