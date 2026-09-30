import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("train.csv")

# print(data["Sex"].unique())
# print(data["Embarked"].unique())

#Cleanup 
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])#for embarked we use most common value (mode)

data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data = pd.get_dummies(data, columns=["Embarked"], dtype=int)


#Assigning data
X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
          "Embarked_C", "Embarked_Q", "Embarked_S"]]
Y = data["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42,
    stratify = Y
)

#Model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))