
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -------------------------
# 1. Create dataset
# -------------------------

data = pd.DataFrame({
    "year": [
        2018, 2020, 2016, 2019, 2021,
        2017, 2022, 2015, 2020, 2019,
        2018, 2021, 2017, 2022, 2016,
        2019, 2020, 2018, 2021, 2017
    ],

    "km_driven": [
        45000, 30000, 80000, 40000, 20000,
        65000, 15000, 95000, 35000, 50000,
        55000, 25000, 70000, 12000, 85000,
        42000, 28000, 60000, 18000, 75000
    ],

    "engine_cc": [
        1197, 1498, 1197, 1498, 1998,
        1197, 1498, 998, 1498, 1197,
        1498, 1998, 1197, 1498, 998,
        1498, 1197, 1998, 1498, 1197
    ],

    "mileage": [
        18.0, 19.5, 17.0, 18.5, 16.0,
        19.0, 20.0, 21.0, 19.2, 18.0,
        18.8, 16.5, 17.5, 20.5, 21.5,
        19.0, 18.2, 16.8, 19.7, 17.2
    ],

    "owners": [
        1, 1, 2, 1, 1,
        2, 1, 3, 1, 2,
        1, 1, 2, 1, 3,
        1, 1, 2, 1, 2
    ],

    "price": [
        720000, 1050000, 480000, 850000, 1250000,
        600000, 1400000, 350000, 1100000, 780000,
        700000, 1300000, 520000, 1500000, 400000,
        900000, 1150000, 650000, 1350000, 500000
    ]
})


# -------------------------
# 2. Separate features and target
# -------------------------

X = data[
    [
        "year",
        "km_driven",
        "engine_cc",
        "mileage",
        "owners"
    ]
]

Y = data["price"]


# -------------------------
# 3. Split data
# -------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# 4. Create and train model
# -------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, Y_train)


# -------------------------
# 5. Make predictions
# -------------------------

predictions = model.predict(X_test)


# -------------------------
# 6. Evaluate model
# -------------------------

mae = mean_absolute_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print("MAE:", mae)
print("R2:", r2)


# -------------------------
# 7. Compare actual vs predicted
# -------------------------

print("\nActual vs Predicted")
print("-------------------------")

for actual, predicted in zip(Y_test, predictions):
    print(
        "Actual:", actual,
        "| Predicted:", round(predicted)
    )

