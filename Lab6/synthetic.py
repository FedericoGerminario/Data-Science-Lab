import pandas as pd
from matplotlib.pyplot import scatter
import matplotlib as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

fp = pd.read_csv("synthetic.csv")
colors = ['r', 'b']
x0 = np.array(fp.iloc[:, 0])
x1 = np.array(fp.iloc[:, 1])
y = np.array(fp.iloc[:, 2])
colors = []
for el in y:
    if el == 0:
        colors.append('r')
    else:
        colors.append('b')

scatter(x0, x1, c=colors)
x_train = np.array(fp.iloc[:, 0:2])

clf = DecisionTreeClassifier()
clf.fit(fp[['x0', 'x1']], y)
dot_code = export_graphviz(clf, feature_names=['x0', 'x1'])
graph = pydot.graph_from_dot_data(dot_code)


print("Finito")

