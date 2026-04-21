import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = r"D:/Dataset/binary_classification_quic_balanced.csv"

FEATURES = [
    'flow_duration',
    'total_fwd_bytes',
    'total_bwd_bytes',
    'total_pkts',
    'total_fwd_pkts',
    'total_bwd_pkts',
    'bytes_per_sec',
    'pkts_per_sec',
]

TARGET = 'label'

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_FILE)

X = df[FEATURES].astype(float)
y = df[TARGET].astype(int)

# -----------------------------
# TRAIN/VALID/TEST SPLIT
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -----------------------------
# BUILD MODEL
# -----------------------------
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_val, y_val)
)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------
# MANUAL PERMUTATION FEATURE IMPORTANCE
# ---------------------------------------------
print("\nCalculating Feature Importance (Manual Permutation Importance)...")


def permutation_importance_keras(model, X_test, y_test, feature_names):
    X_test_copy = X_test.copy()

    # Baseline accuracy
    baseline_preds = (model.predict(X_test_copy) > 0.5).astype(int)
    baseline_acc = (baseline_preds.flatten() == y_test.values).mean()

    importances = []

    for i, feature in enumerate(feature_names):
        X_permuted = X_test_copy.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])  # shuffle feature

        perm_preds = (model.predict(X_permuted) > 0.5).astype(int)
        perm_acc = (perm_preds.flatten() == y_test.values).mean()

        importance = baseline_acc - perm_acc
        importances.append(importance)

        print(f"Feature: {feature:20s} Importance (Drop in Accuracy): {importance:.6f}")

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)


importance_df = permutation_importance_keras(
    model,
    X_test,
    y_test,
    FEATURES
)

print("\n===== FINAL FEATURE IMPORTANCE =====")
print(importance_df)

# -----------------------------
# PLOT FEATURE IMPORTANCE (Horizontal)
# -----------------------------
plt.figure(figsize=(10, 6))

importance_df_sorted = importance_df.sort_values(by="importance", ascending=True)

plt.barh(
    importance_df_sorted["feature"],
    importance_df_sorted["importance"]
)

plt.xlabel("Importance (Accuracy Drop)")
plt.ylabel("Features")
plt.title("Neural Network Feature Importance (Permutation)")

# Add value labels
for index, value in enumerate(importance_df_sorted["importance"]):
    plt.text(
        value,
        index,
        f"{value:.4f}",
        va='center',
        ha='left'
    )

plt.tight_layout()
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("D:/neural_network_quic_model.h5")
print("\nModel saved.")
