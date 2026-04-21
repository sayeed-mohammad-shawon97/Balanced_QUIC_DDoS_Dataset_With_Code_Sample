import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = r"D:/Dataset/binary_classification_quic_balanced.csv"
MODEL_FILE = r"D:/catboost_quic_model.cbm"

FEATURES = [
    'flow_duration',
    'total_fwd_bytes',
    'total_bwd_bytes',
    'total_pkts',
    'total_fwd_pkts',
    'total_bwd_pkts',
    'bytes_per_sec',
    'pkts_per_sec',
    #'fwd_bwd_ratio'
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

# Convert features to float if necessary
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
# CREATE CATBOOST MODEL
# -----------------------------
model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='F1',
    iterations=1000,
    learning_rate=0.003,
    depth=8,
    l2_leaf_reg=3,
    random_strength=1,
    border_count=128,
    verbose=100,
    random_seed=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100
)

# -----------------------------
# EVALUATE
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# FEATURE IMPORTANCE PLOT
# -----------------------------
feature_importances = model.get_feature_importance(Pool(X_train, y_train))
plt.figure(figsize=(10,6))
plt.barh(FEATURES, feature_importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()  # highest importance on top
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_model(MODEL_FILE)
print(f"\nModel saved to: {MODEL_FILE}")
