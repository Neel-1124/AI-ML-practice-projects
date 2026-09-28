import pandas as pd

data = pd.read_csv("train.csv")

print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.info )