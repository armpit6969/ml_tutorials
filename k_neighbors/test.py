import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
# best = 0
# for _ in range(1000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     model = KNeighborsClassifier(n_neighbors=9)
#
#     model.fit(x_train, y_train)
#     acc = model.score(x_test, y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         with open("studentgrades.pickle", "wb") as f:
#             pickle.dump(model, f)
# print("Best Accuracy: ", best)


# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
model = pickle.load(pickle_in)

acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# for x in range(len(predicted)):
#     print("Predicted: ", names[predicted[x]])
#     print("Data: ", x_test[x])
#     print("Actual: ", names[y_test[x]])
#     # n = model.kneighbors([x_test[x]], 9, True)
#     # print("N: ", n)