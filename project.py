import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#uruchomienie -> w terminalu: python project.py


# Wczytaliśmy 30 000 rekordów klientów banku i usunęliśmy kolumnę ID — symulacja usuwania danych wrażliwych
df = pd.read_csv("UCI_Credit_Card.csv")
#print(df.shape) #wyswietla (wiersze, kolumny)
#print(df.head()) # wyswietla pierwsze 5 wierszy

df = df.drop(columns=["ID"]) #usuwa kolumne "ID"
#print(df.columns.tolist()) #sprawdzenie czy zniknelo



# Podzieliliśmy dane na dwa niezależne oddziały banku po 15 000 klientów. Oddział A nie widzi danych B i odwrotnie
half = len(df) // 2 #liczy polowe wierszy
tenant_A = df.iloc[:half] #bierze pierwsza polowe
tenant_B = df.iloc[half:] #bierze druga polowe



# Rozdzielenie X i y: X = dane o kliencie (wiek, limit, spłaty...), y = czy spłacił kredyt (0 lub 1). Robimy to dla obu oddziałów
X_A = tenant_A.drop(columns=["default.payment.next.month"]) #dane wejsciowe - informacje o kliencie
y_A = tenant_A["default.payment.next.month"] #zmienna docelowa - jedna kolumna ktora mowi czy klient splacil kredyt

X_B = tenant_B.drop(columns=["default.payment.next.month"])
y_B = tenant_B["default.payment.next.month"]

#print(f"Tenant_A: {X_A.shape}, y_A: {y_A.shape}") #wypisanie
#print(f"Tenant_B: {X_B.shape}, y_B: {y_B.shape}")



# Każdy oddział trenuje swój model osobno na swoich danych. Skalujemy dane żeby model działał poprawnie. Na końcu oddział zwraca tylko wagi — nie dane klientów
def trenuj_lokalnie(X, y, noise_level):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # skaluje dane do podobnego zakresu

    model = LogisticRegression(max_iter=1000) # model który uczy się na danych oddziału. Mimo nazwy "regresja" służy do klasyfikacji (tu: czy klient spłaci kredyt czy nie), max 1000 krokow podczas nauki
    model.fit(X_scaled, y) # właściwe "uczenie się" — model analizuje dane i wylicza wagi
    
    wagi = model.coef_        # wagi dla każdej z 23 kolumn. Mówią jak bardzo dana cecha (np. wiek, limit) wpływa na wynik
    bias = model.intercept_   #  bias, czyli punkt startowy modelu niezależny od danych

    # Dodanie szumu do wag przed wysłaniem do centrali
    szum_wagi = np.random.normal(0, noise_level, wagi.shape)
    szum_bias = np.random.normal(0, noise_level, bias.shape)
    
    wagi_z_szumem = wagi + szum_wagi
    bias_z_szumem = bias + szum_bias
    
    return wagi_z_szumem, bias_z_szumem


noise_level = 0.0

# Trenowanie każdego oddziału osobno
wagi_A, bias_A = trenuj_lokalnie(X_A, y_A, noise_level)
wagi_B, bias_B = trenuj_lokalnie(X_B, y_B, noise_level)

print("Wagi oddziału A:", wagi_A)
print("Bias oddziału A:", bias_A)
print("Wagi oddziału B:", wagi_B)
print("Bias oddziału B:", bias_B)



# Centrala zbiera wagi od obu oddziałów i uśrednia je. To jest serce Federated Learning — centrala nigdy nie widziała danych CSV, tylko liczby
def fedavg(wagi_A, bias_A, wagi_B, bias_B):
    wagi_globalne = (wagi_A + wagi_B) / 2  # uśrednienie wag
    bias_globalny = (bias_A + bias_B) / 2  # uśrednienie biasu
    return wagi_globalne, bias_globalny

# Agregacja
wagi_globalne, bias_globalny = fedavg(wagi_A, bias_A, wagi_B, bias_B)



# Tworzymy pusty model i wstrzykujemy do niego uśrednione wagi. Model globalny ma wiedzę z obu oddziałów — bez dostępu do ich danych
model_globalny = LogisticRegression()
model_globalny.coef_ = wagi_globalne        # wstrzykujemy wagi
model_globalny.intercept_ = bias_globalny  # wstrzykujemy bias
model_globalny.classes_ = np.array([0, 1]) # mówimy modelowi jakie są możliwe wyniki

print("Wagi globalne:", wagi_globalne)
print("Bias globalny:", bias_globalny)
print("Model globalny gotowy!")






# Zbiór testowy — osobne dane do oceny modelu
X_all = df.drop(columns=["default.payment.next.month"])
y_all = df["default.payment.next.month"]

scaler_testowy = StandardScaler()
X_all_scaled = scaler_testowy.fit_transform(X_all)

_, X_test, _, y_test = train_test_split(X_all_scaled, y_all, test_size=0.2, random_state=42)

# Pętla eksperymentalna
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
dokladnosci = []

for noise in noise_levels:
    wagi_A, bias_A = trenuj_lokalnie(X_A, y_A, noise)
    wagi_B, bias_B = trenuj_lokalnie(X_B, y_B, noise)
    
    wagi_glob, bias_glob = fedavg(wagi_A, bias_A, wagi_B, bias_B)
    
    model = LogisticRegression()
    model.coef_ = wagi_glob
    model.intercept_ = bias_glob
    model.classes_ = np.array([0, 1])
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    dokladnosci.append(acc)
    print(f"noise_level={noise:.2f} → Dokładność: {acc*100:.1f}%")

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, dokladnosci, marker="o", color="black", linewidth=2)
plt.title("Cena Prywatności — Trade-off szum vs dokładność")
plt.xlabel("Poziom szumu (noise_level)")
plt.ylabel("Dokładność modelu globalnego")
plt.grid(True)
plt.savefig("wykres.png")
plt.show()
print("Wykres zapisany jako wykres.png")