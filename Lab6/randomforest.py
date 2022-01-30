from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class MyRandomForestClassifier:
    def __init__(self, n_estimators, max_features):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        indeces = np.arange(0, len(X))
        for i in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_features=self.max_features)
            x_train_indices = np.random.choice(indeces, len(X), replace=True)
            clf.fit(X[x_train_indices], y[x_train_indices])
            self.trees.append(clf)

    def predict(self, x):
        y_predict = pd.DataFrame()
        for i, clf in enumerate(self.trees):
            y_predict[i] = (clf.predict(x))

        return y_predict.mode(axis=1)[0]

    def getParametersImportance(self):
        importance = pd.DataFrame()
        for i, clf in enumerate(self.trees):
            importance[i]=clf.feature_importances_

        rows = importance.sum(axis=1)
        total = rows.sum()
        return (rows/total)

dataset = fetch_openml("mnist_784")
X = dataset['data']
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print("Accuracy ", accuracy_score(y_test, y_predict))

my_forest = MyRandomForestClassifier(10, 'sqrt')
my_forest.fit(x_train, y_train)
my_y_predict_forest = my_forest.predict(x_test)

forest = RandomForestClassifier(n_estimators=10, max_features= 'sqrt')
forest.fit(x_train, y_train)
y_predict_forest = forest.predict(x_test)


print("MyForest accuracy: ", accuracy_score(y_test, my_y_predict_forest))
print("Forest accuracy: ", accuracy_score(y_test, y_predict_forest))
importance = np.array(my_forest.getParametersImportance())

#Check if the total value is 1.0

print(importance.sum())
#Part 7

#my forest feature importance heatmap
features = np.reshape(importance, (28, 28))
sns.heatmap(features, cmap='binary')

#sklearn forest feature importance heatmap
importance_forest = forest.feature_importances_
features_forest = np.reshape(importance_forest, (28, 28))
plt.figure(2)
sns.heatmap(features_forest, cmap='binary')


print('finito')
