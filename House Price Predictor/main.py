import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv("train.csv", na_values=["", " ", "NA", "N/A"])

missing_value = data.isnull().sum()
None_columns = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature"
]
data[None_columns] = data[None_columns].fillna("None")

data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0)
data["Electrical"] = data["Electrical"].fillna(data["Electrical"].mode()[0])

X = data[[
    "OverallQual",   
    "GrLivArea",     
    "GarageCars",    
    "GarageArea",    
    "TotalBsmtSF",   
    "1stFlrSF",      
    "FullBath",      
    "YearBuilt",     
    "YearRemodAdd",  
    "TotRmsAbvGrd",  
    "Fireplaces",    
    "GarageYrBlt",   
    "MasVnrArea",    
    "LotArea",
    "Neighborhood",
    "KitchenQual"
]]

Y = data["SalePrice"]

X = pd.get_dummies(X, dtype= int)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42
)

model = RandomForestRegressor(
    n_estimators= 100,
    random_state= 42
)
model.fit(X_train, Y_train)
Predictions = model.predict(X_test)

print(mean_absolute_error(Y_test, Predictions))
print(root_mean_squared_error(Y_test, Predictions))
print(r2_score(Predictions, Y_test))

plt.scatter(Y_test, Predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predictions")

plt.show()