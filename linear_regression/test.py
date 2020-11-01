import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("student-mat.csv", sep=";")

# Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:
predict = "G3"

data = data[["Dalc", "studytime", "freetime", "failures", "absences", "G1", "G2", "G3"]]
data = sklearn.utils.shuffle(data) # Optional - shuffle the data

le = preprocessing.LabelEncoder()
# famsup = le.fit_transform(list(data["famsup"]))
# schoolsup = le.fit_transform(list(data["schoolsup"]))
# paid = le.fit_transform(list(data["paid"]))
# higher = le.fit_transform(list(data["higher"]))
# internet = le.fit_transform(list(data["internet"]))


# data.update({"famsup": famsup, "schoolsup": schoolsup, "paid": paid, "higher": higher, "internet": internet})

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
# best = 0
# for _ in range(1000):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: " + str(acc))
#
#     if acc > best:
#         best = acc
#         with open("studentgrades.pickle", "wb") as f:
#             pickle.dump(linear, f)
# print("Best Accuracy: ", best)


# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


acc = linear.score(x_test, y_test)
print("Accuracy: " + str(acc))

# print("-------------------------")
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
# print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# # Drawing and plotting model
# plot = "failures"
# plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()