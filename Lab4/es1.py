import numpy as np
import pandas as pd

class KNearestNeighbors:
    def __init__(self, k, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.x_train = None
        self.y_train = None
        self.weights = weights

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):

        predictions = []

        for i, p in enumerate(x[:]):
            distances = []
            for en, q in enumerate(self.x_train[:]):
                currentDistance = self.weightdis[self.weights](self, p=p, q=q)
                distances.append(currentDistance)

            values = np.array(distances)
            indexes = np.argsort(values)
            kNearestIndexes = indexes[:(self.k - 1)]
            distance = {el: 0 for el in set(self.y_train)}
            for el in kNearestIndexes:
                distance[y_train[el]] += 1
            #Voting process
            predictions.append(max(distance, key=distance.get))


        return np.array(predictions)



    def weightedDistance(self, p, q):
        return 1/(self.distance[self.distance_metric](self, p, q))

    def uniformDistance(self, p, q):
        return self.distance[self.distance_metric](self, p, q)

    def euclideanDistance(self, x, y):
        return ((x - y)**2).sum()

    def cosineDistance(self, x, y):
        return 1 - abs(((x*y).sum())/((((x**2).sum())**0.5)*(((y**2).sum())**0.5)))

    def manhattanDistance(self, x, y):
        return np.sum(abs(x - y))


    distance = {
        'euclidean': euclideanDistance,
        'cosine': cosineDistance,
        'manhattan': manhattanDistance
    }
    weightdis = {
        'distance': weightedDistance,
        'uniform': uniformDistance
    }


def predict(x):

    prediction = []
    for i, el in enumerate(x[:]):
        print(i)
        print(el)

def euclideanDistance(x, y):
        return ((x - y)**2).sum()
'''
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[11,12,13],[ 14, 15, 16]])
#z = np.concatenate((x, y))
#print(z)

v = np.vstack((x, y))
print(v)

z = np.array([7, 7, 9, 9, 8, 8])
print(np.split(z, [2, 4]))

s = pd.Series([1, 2.5, 3.4, 5], index=['mon','tue', 'wed', 'thur'])
s.loc['tue']= 10
s.iloc[0] = 10
isok = s >= 5
'''

df = pd.read_csv('iris.csv', header=None)
df1 = df.sample(frac=0.2, replace=False)
x_test = np.array(df1.iloc[:, 0:4])
y_test = np.array(df1.iloc[:, 4])
x_train = df.iloc[df.index.difference(df1.index)].iloc[:, 0:4].values
y_train = df.iloc[df.index.difference(df1.index)].iloc[:, 4].values
'''
dfMnist = pd.read_csv('mnist.csv', header=None)
print(dfMnist)
dfMnistColumnToSample = dfMnist.iloc[:, 1:]
dfMnistIndex = dfMnist.iloc[:, 0]
dfMnistColumn = dfMnistColumnToSample.sample(n=100, axis='columns')

df1Mnist = dfMnistColumn.sample(n=1000, replace=False)
x_testMnist = np.array(df1Mnist)

x_trainMnist = dfMnistColumn.iloc[dfMnistColumn.index.difference(df1Mnist.index)].values
y_trainMnist = dfMnistIndex.iloc[dfMnistIndex.index.difference(df1Mnist.index)].values

print(x_trainMnist)

knn = KNearestNeighbors(1, 'euclidean', 'uniform')
knn.fit(x_trainMnist, y_trainMnist)
print(knn.predict(x_testMnist))
'''
knn = KNearestNeighbors(5, 'euclidean', 'uniform')
knn.fit(x_train, y_train)
print(knn.predict(x_test))









'''
predictions = []
for i, p in enumerate(x_test[:]):
    distance = {el: 0 for el in set(y_train)}
    for en, q in enumerate(x_train[:]):
        if euclideanDistance(p, q) <= 3:
            distance[y_train[en]] += 1
    predictions.append(max(distance, key=distance.get))

print(np.array(predictions))
#print(x_test[:][0])
'''
