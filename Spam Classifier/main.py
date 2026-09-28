import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

data = pd.read_csv(
    "SMSSpamCollection",
    sep = "\t",
    header = None,
    names = ["label", "message"]
)

#Data Cleanup
data["message"] = data["message"].str.lower()
data = data.drop_duplicates()
data = data.reset_index(drop=True)

#This is a commit test

#Test Train Split
X = data["message"]
Y = data["label"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42,
    stratify = Y
)

#Vectorizing
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

#LogisticRegression Model
# model = LogisticRegression(max_iter= 1000)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_test)
# CM = confusion_matrix(Y_test, predictions)
# print(CM)

#Bayes Theorem
model = MultinomialNB()

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

CM = confusion_matrix(Y_test, predictions)

print(CM)