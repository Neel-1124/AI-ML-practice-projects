import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#Cleanup
data["TotalCharges"] = data["TotalCharges"].str.strip()
data["TotalCharges"] = data["TotalCharges"].replace("", 0)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

#Assigning
X = data[[
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
    ]]

Y = data["Churn"]

#Encoding
X = pd.get_dummies(X, dtype=int)
Y = Y.map({"No": 0, "Yes": 1})

#Training Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

 #Model Testing
model_LR = LogisticRegression(max_iter=1000)
model_LR.fit(X_train, Y_train)
Predictions_LR = model_LR.predict(X_test)

 #LR Accuracy
print(accuracy = accuracy_score(Predictions_LR, Y_test))
print(LR_confusion = confusion_matrix(Predictions_LR, Y_test))
print(classification_report(Y_test, Predictions_LR))








#Random Forest Testing

# model = RandomForestClassifier()
# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)


# print(accuracy_score(predictions, Y_test))
# print(confusion_matrix(predictions, Y_test))

# print(classification_report(predictions, Y_test))


   #Decision Tree method

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# Predictions = model.predict(X_test)

# accuracy = accuracy_score(Predictions, Y_test)
# print(accuracy)

# Confusion = confusion_matrix(Predictions, Y_test)
# print(Confusion)

# print(classification_report(Y_test, Predictions))