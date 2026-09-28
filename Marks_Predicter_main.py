import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

student_data = pd.DataFrame({
    "study_hours": [
        2, 3, 1.5, 4, 5, 2.5, 6, 7, 3.5, 8,
        4.5, 1, 6.5, 5.5, 2, 7.5, 9, 3, 4, 8.5,
        5, 6, 2.5, 7, 3.5, 9.5, 4, 5.5, 8, 6.5
    ],

    "attendance": [
        52, 61, 48, 72, 78, 55, 85, 91, 64, 94,
        76, 43, 88, 82, 58, 89, 97, 62, 70, 93,
        74, 86, 51, 90, 67, 96, 73, 80, 92, 84
    ],

    "previous_marks": [
        45, 52, 38, 65, 72, 48, 81, 88, 57, 91,
        69, 35, 84, 76, 50, 86, 95, 55, 63, 89,
        71, 79, 42, 87, 60, 94, 68, 74, 90, 77
    ],

    "practice_tests": [
        1, 2, 0, 3, 4, 1, 5, 6, 2, 7,
        4, 0, 5, 4, 1, 6, 8, 2, 3, 7,
        4, 5, 1, 6, 3, 8, 3, 5, 7, 5
    ],

    "passed": [
        0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
        1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
        1, 1, 0, 1, 0, 1, 1, 1, 1, 1
    ]
})

new_student = [[6, 82, 70, 5]]

X = student_data[[
    "study_hours",
    "attendance",
    "previous_marks",
    "practice_tests"
]]

Y = student_data["passed"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42
)

year = int(input("Enter year: "))
mileage = int(input("Enter milage: "))
engine = int(input("Enter engine size: "))

new_car = [[year, mileage, engine]]

#__________________________________________________________


print(X_train.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)

print("__________________________")

model = LogisticRegression()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(predictions)

print("__________________________")

accuracy = accuracy_score(Y_test, predictions)
print ("Accuracy:", accuracy)

print("__________________________")

cm = confusion_matrix(Y_test, predictions)
print(cm)

print("__________________________")

predictions = model.predict(new_car)
print(predictions)