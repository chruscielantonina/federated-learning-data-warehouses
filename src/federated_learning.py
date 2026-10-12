import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# w terminalu -> pip install pandas scikit-learn numpy matplotlib
# uruchomienie -> w terminalu: python project.py


np.random.seed(42)

df = pd.read_csv("UCI_Credit_Card.csv")
df = df.drop(columns=["ID"])


print("Braki danych (NaN) łącznie:", df.isna().sum().sum())
print("EDUCATION przed czyszczeniem:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE przed czyszczeniem:", sorted(df["MARRIAGE"].unique()))

df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("EDUCATION po czyszczeniu:", sorted(df["EDUCATION"].unique()))
print("MARRIAGE po czyszczeniu:", sorted(df["MARRIAGE"].unique()))


baseline_acc = (df["default.payment.next.month"] == 0).mean()
print(f"\nBaseline (predykcja samych zer): {baseline_acc*100:.1f}%")
print(f"Rozkład klas: {df['default.payment.next.month'].value_counts().to_dict()}\n")



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


def trenuj_lokalnie(X_scaled, y, noise_level):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    wagi = model.coef_
    bias = model.intercept_

    szum_wagi = np.random.normal(0, noise_level, wagi.shape)
    szum_bias = np.random.normal(0, noise_level, bias.shape)

    wagi_z_szumem = wagi + szum_wagi
    bias_z_szumem = bias + szum_bias

    return wagi_z_szumem, bias_z_szumem


def fedavg(wagi_A, bias_A, wagi_B, bias_B):
    wagi_globalne = (wagi_A + wagi_B) / 2
    bias_globalny = (bias_A + bias_B) / 2
    return wagi_globalne, bias_globalny


def zbuduj_model_globalny(wagi, bias):
    model = LogisticRegression()
    model.coef_ = wagi
    model.intercept_ = bias
    model.classes_ = np.array([0, 1])
    return model


noise_level = 0.0
wagi_A, bias_A = trenuj_lokalnie(X_A_scaled, y_A, noise_level)
wagi_B, bias_B = trenuj_lokalnie(X_B_scaled, y_B, noise_level)

print("Wagi oddziału A:", wagi_A)
print("Bias oddziału A:", bias_A)
print("Wagi oddziału B:", wagi_B)
print("Bias oddziału B:", bias_B)

wagi_globalne, bias_globalny = fedavg(wagi_A, bias_A, wagi_B, bias_B)
print("Wagi globalne:", wagi_globalne)
print("Bias globalny:", bias_globalny)
print("Model globalny gotowy!\n")



noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
dokladnosci = []

for noise in noise_levels:
    wagi_A, bias_A = trenuj_lokalnie(X_A_scaled, y_A, noise)
    wagi_B, bias_B = trenuj_lokalnie(X_B_scaled, y_B, noise)

    wagi_glob, bias_glob = fedavg(wagi_A, bias_A, wagi_B, bias_B)
    model = zbuduj_model_globalny(wagi_glob, bias_glob)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    dokladnosci.append(acc)
    print(f"noise_level={noise:.2f} -> Dokladnosc: {acc*100:.1f}% (baseline: {baseline_acc*100:.1f}%)")

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, dokladnosci, marker="o", color="black", linewidth=2, label="Model FedAvg")
plt.axhline(y=baseline_acc, color="red", linestyle="--", linewidth=1,
            label=f"Baseline (zawsze 0) = {baseline_acc*100:.1f}%")
plt.title("Cena Prywatności - Trade-off szum vs dokładność")
plt.xlabel("Poziom szumu (noise_level)")
plt.ylabel("Dokładność modelu globalnego")
plt.legend()
plt.grid(True)
plt.savefig("wykres.png")
plt.show()
print("Wykres zapisany jako wykres.png")
