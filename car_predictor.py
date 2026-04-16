import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load():
    d1=pd.read_csv("car data.csv")
    df=pd.DataFrame(d1)
    df["age"]=2026-df["Year"]
    df.drop("Year",axis=1,inplace=True)
    df.drop("Car_Name",axis=1,inplace=True)
    df_encode=pd.get_dummies(df,drop_first=True,columns=["Fuel_Type","Seller_Type","Transmission"])
    return df_encode
def train(df_encode):
    X=df_encode.drop("Selling_Price",axis=1)
    y=df_encode["Selling_Price"]
    m = LinearRegression()
    m.fit(X, y)
    return m
    
data=load()
t_m=train(data)
print("--- AI Car Price Predictor ---")
p = float(input("Present Price (Lakhs): "))
k = int(input("Kms Driven: "))
o = int(input("Owner (0/1/3): "))
a = int(input("Age of car: "))

# Turning words into the 1/0 numbers the model needs
f = input("Fuel (Petrol/Diesel): ").capitalize()
d = 1 if f == "Diesel" else 0
pe = 1 if f == "Petrol" else 0

s = input("Seller (Individual/Dealer): ").capitalize()
i = 1 if s == "Individual" else 0

t = input("Transmission (Manual/Automatic): ").capitalize()
m = 1 if t == "Manual" else 0

# The "2D Array" fix: [[...]]
features = [[p, k, o, a, d, pe, i, m]]
prediction = t_m.predict(features)

print(f"\nPredicted Resale Value: {prediction[0]:.2f} Lakhs")