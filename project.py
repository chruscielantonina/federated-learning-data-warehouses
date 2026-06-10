import pandas as pd
from sklearn.model_selection import train_test_split

#uruchomienie -> w terminalu: python project.py
df = pd.read_csv("UCI_Credit_Card.csv")  # plik w tym samym folderze
print(df.shape) #wyswietla (wiersze, kolumny)
print(df.head()) # wyswietla pierwsze 5 wierszy

df = df.drop(columns=["ID"]) #usuwa kolumne "ID"
print(df.columns.tolist()) #sprawdzenie czy zniknelo

half = len(df) // 2 #liczy polowe wierszy
tenant_A = df.iloc[:half] #bierze pierwsza polowe
tenant_B = df.iloc[half:] #bierze druga polowe

X_A = tenant_A.drop(columns=["default.payment.next.month"]) #dane wejsciowe - informacje o kliencie
y_A = tenant_A["default.payment.next.month"] #zmienna docelowa - jedna kolumna ktora mowi czy klient splacil kredyt

X_B = tenant_B.drop(columns=["default.payment.next.month"])
y_B = tenant_B["default.payment.next.month"]

print(f"Tenant_A: {X_A.shape}, y_A: {y_A.shape}") #wypisanie
print(f"Tenant_B: {X_B.shape}, y_B: {y_B.shape}")
