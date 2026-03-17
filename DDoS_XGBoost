
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = r"D:/Dataset/binary_classification_quic_balanced.csv"
MODEL_FILE = r"D:/xgboost_quic_model.json"

FEATURES = [
    'flow_duration',
    'total_fwd_bytes',
    'total_bwd_bytes',
    'total_pkts',
    'total_fwd_pkts',
    'total_bwd_pkts',
    'bytes_per_sec',
    'pkts_per_sec',
    # 'fwd_bwd_ratio'
]
TARGET = 'label'

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_FILE)

# Ensure all features exist
for f in FEATURES + [TARGET]:
    if f not in df.columns:
        raise ValueError(f"Missing column: {f}")

X = df[FEATURES].astype(float)
y = df[TARGET]

# -----------------------------
# TRAIN/VALIDATION/TEST SPLIT
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training size: {len(X_train)} | Class balance:\n{y_train.value_counts()}")
print(f"Validation size: {len(X_val)} | Class balance:\n{y_val.value_counts()}")
print(f"Test size: {len(X_test)} | Class balance:\n{y_test.value_counts()}")

# -----------------------------
# -----------------------------
# CREATE XGBOOST MODEL
# -----------------------------
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=1000,
    learning_rate=0.003,
    max_depth=8,
    reg_lambda=3,
    random_state=42,
    use_label_encoder=False
)

# ✅ FIX — set early stopping here
model.set_params(early_stopping_rounds=100)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


# -----------------------------
# EVALUATE
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# -----------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------
feature_importances = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importances
})

# Sort by importance (descending)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display as table
print("\nXGBoost Feature Importance Table:")
print(importance_df.to_string(index=False))


# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_model(MODEL_FILE)
print(f"\nModel saved to: {MODEL_FILE}")
