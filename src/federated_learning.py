import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

# Install: pip install -r requirements.txt
# Run (from the repository root): python src/federated_learning.py

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "UCI_Credit_Card.csv"
PLOT_PATH = ROOT / "results" / "noise_vs_accuracy.png"


np.random.seed(42)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["ID"])


print("Missing values (NaN) in total:", df.isna().sum().sum())
print("EDUCATION before cleaning:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE before cleaning:", sorted(df["MARRIAGE"].unique()))

df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("EDUCATION after cleaning:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE after cleaning:", sorted(df["MARRIAGE"].unique()))


baseline_acc = (df["default.payment.next.month"] == 0).mean()
print(f"\nBaseline (always predicting 0): {baseline_acc*100:.1f}%")
print(f"Class distribution: {df['default.payment.next.month'].value_counts().to_dict()}\n")



X_all = df.drop(columns=["default.payment.next.month"])
y_all = df["default.payment.next.month"]

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

half = len(X_train_pool) // 2
X_A = X_train_pool.iloc[:half]
y_A = y_train_pool.iloc[:half]
X_B = X_train_pool.iloc[half:]
y_B = y_train_pool.iloc[half:]



scaler = StandardScaler()
scaler.fit(X_train_pool)

X_A_scaled = scaler.transform(X_A)
X_B_scaled = scaler.transform(X_B)
X_test_scaled = scaler.transform(X_test)


def train_local(X_scaled, y, noise_level):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    weights = model.coef_
    bias = model.intercept_

    weights_noise = np.random.normal(0, noise_level, weights.shape)
    bias_noise = np.random.normal(0, noise_level, bias.shape)

    noisy_weights = weights + weights_noise
    noisy_bias = bias + bias_noise

    return noisy_weights, noisy_bias


def fedavg(weights_a, bias_a, weights_b, bias_b):
    global_weights = (weights_a + weights_b) / 2
    global_bias = (bias_a + bias_b) / 2
    return global_weights, global_bias


def build_global_model(weights, bias):
    model = LogisticRegression()
    model.coef_ = weights
    model.intercept_ = bias
    model.classes_ = np.array([0, 1])
    return model


noise_level = 0.0
weights_a, bias_a = train_local(X_A_scaled, y_A, noise_level)
weights_b, bias_b = train_local(X_B_scaled, y_B, noise_level)

print("Branch A weights:", weights_a)
print("Branch A bias:", bias_a)
print("Branch B weights:", weights_b)
print("Branch B bias:", bias_b)

global_weights, global_bias = fedavg(weights_a, bias_a, weights_b, bias_b)
print("Global weights:", global_weights)
print("Global bias:", global_bias)
print("Global model ready!\n")



noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
accuracies = []

for noise in noise_levels:
    weights_a, bias_a = train_local(X_A_scaled, y_A, noise)
    weights_b, bias_b = train_local(X_B_scaled, y_B, noise)

    global_weights, global_bias = fedavg(weights_a, bias_a, weights_b, bias_b)
    model = build_global_model(global_weights, global_bias)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"noise_level={noise:.2f} -> Accuracy: {acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%)")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, accuracies, marker="o", color="black", linewidth=2, label="FedAvg global model")
plt.axhline(y=baseline_acc, color="red", linestyle="--", linewidth=1,
            label=f"Baseline (always predict 0) = {baseline_acc*100:.1f}%")
plt.title("Price of Privacy: noise vs. accuracy trade-off")
plt.xlabel("Noise level (noise_level)")
plt.ylabel("Global model accuracy")
plt.legend()
plt.grid(True)
PLOT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {PLOT_PATH}")