import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

titanic_data = pd.read_csv(r"C:\Users\Daddy\Downloads\titanic\train.csv")
attributes = ["Pclass","Parch","Sex","SibSp"]
X = pd.get_dummies(titanic_data[attributes])
Y = titanic_data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
score = accuracy_score(y_test, predictions)
score
