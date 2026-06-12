import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler

#uruchomienie -> w terminalu: python project.py
df = pd.read_csv("UCI_Credit_Card.csv")  # plik w tym samym folderze
#print(df.shape) #wyswietla (wiersze, kolumny)
#print(df.head()) # wyswietla pierwsze 5 wierszy

df = df.drop(columns=["ID"]) #usuwa kolumne "ID"
#print(df.columns.tolist()) #sprawdzenie czy zniknelo

half = len(df) // 2 #liczy polowe wierszy
tenant_A = df.iloc[:half] #bierze pierwsza polowe
tenant_B = df.iloc[half:] #bierze druga polowe

X_A = tenant_A.drop(columns=["default.payment.next.month"]) #dane wejsciowe - informacje o kliencie
y_A = tenant_A["default.payment.next.month"] #zmienna docelowa - jedna kolumna ktora mowi czy klient splacil kredyt

X_B = tenant_B.drop(columns=["default.payment.next.month"])
y_B = tenant_B["default.payment.next.month"]

#print(f"Tenant_A: {X_A.shape}, y_A: {y_A.shape}") #wypisanie
#print(f"Tenant_B: {X_B.shape}, y_B: {y_B.shape}")




def trenuj_lokalnie(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # skaluje dane do podobnego zakresu

    model = LogisticRegression(max_iter=1000) # model który uczy się na danych oddziału. Mimo nazwy "regresja" służy do klasyfikacji (tu: czy klient spłaci kredyt czy nie), max 1000 krokow podczas nauki
    model.fit(X_scaled, y) # właściwe "uczenie się" — model analizuje dane i wylicza wagi
    
    wagi = model.coef_        # wagi dla każdej z 23 kolumn. Mówią jak bardzo dana cecha (np. wiek, limit) wpływa na wynik
    bias = model.intercept_   #  bias, czyli punkt startowy modelu niezależny od danych
    
    return wagi, bias

# Trenowanie każdego oddziału osobno
wagi_A, bias_A = trenuj_lokalnie(X_A, y_A)
wagi_B, bias_B = trenuj_lokalnie(X_B, y_B)

print("Wagi oddziału A:", wagi_A)
print("Bias oddziału A:", bias_A)
print("Wagi oddziału B:", wagi_B)
print("Bias oddziału B:", bias_B)


