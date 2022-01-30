from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pydot
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold



#Part 1
dataset = load_wine()

x = dataset["data"]
y = dataset["target"]
feature_names = dataset["feature_names"]

print(len(x))
print(len(y))
print(len(feature_names))
nul_x = np.isnan(x)
nul_y = np.isnan(y)

unique, counts  = np.unique(y, return_counts=True)
numofvalues = {k:v for k, v in zip(unique, counts)}


print(numofvalues)
if True in nul_x or True in nul_y:
    print("There is at least one null value")

#Part 2 Tree creation
clf = DecisionTreeClassifier()
clf = clf.fit(x, y)

#Part 3 Tree visualization
#first approach using plot.tree()

#plot_tree(clf)
#plt.show()

#second approach using export_graphviz()

dot_code = export_graphviz(clf, feature_names=feature_names)
graph = pydot.graph_from_dot_data(dot_code)
#currently the following piece of code is not working
#Image(graph[0].create_png())

print(dot_code)

#Part 4
y_predict = clf.predict(x)


print('Initial accuracy:', accuracy_score(y, y_predict))

print("target", y)
#Part 5

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Part 6
clf2 = DecisionTreeClassifier()
clf2.fit(x_train, y_train)
y_predict2 = clf2.predict(x_test)
print('Accuracy is: ', accuracy_score(y_test, y_predict2))
clas=['0', '1', '2']
report = classification_report(y_test, y_predict2, target_names=clas)
print(report)

#Part 7
params = {
    "max_depth": [None, 2, 4, 8],
    "splitter": ["best", "random"],
    "criterion": ["gini", "entropy"],
    "max_features":["auto", "sqrt", "log2"]
}
'''
for config in ParameterGrid(params):
    clf3 = DecisionTreeClassifier(**config)
    clf3.fit(x_train, y_train)
    y_predict3 = clf3.predict(x_test)
    print("Configuragion", config)
    print("Accuracy score", accuracy_score(y_test, y_predict3))
'''
#Part 8

X_train_valid, X_test, y_train_valid, y_test = train_test_split(x, y, test_size=0.2)
kf = KFold(5)

for train_indices, validation_indices in kf.split(X_train_valid):
    X_train = X_train_valid[train_indices]
    X_valid = X_train_valid[validation_indices]
    Y_train = y_train_valid[train_indices]
    Y_valid = y_train_valid[validation_indices]
    clf4 = DecisionTreeClassifier()
    clf4.fit(X_train, Y_train)
    Y_predict = clf4.predict(X_valid)
    print("classifier: ", accuracy_score(Y_valid, Y_predict))
    print(clf4.tree_.feature)
    print(clf4.tree_.impurity)

