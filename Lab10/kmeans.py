import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Kmeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.current_iter = 0

    def first_centroids(self, x):
        points = np.random.randint(0, len(x), min(n_clusters, len(x)))
        self.centroids = x[points]
        return self.centroids

    def fit_predict(self, x, plot_clusters=False, plot_step=5):
        #calculate first centroids
        current_centroid = self.first_centroids(x)

        for j in range(self.max_iter):
            if plot_clusters == True:
                if j % plot_step == 0:
                    print("Plotting...")
                    self.plotCluster(x)

            clusters = []
            toCentroid = []
            for el in range(n_clusters):
                toCentroid.append([])

            for i in range(len(x)):
                distance = [self.euclidean_function(x[i], el) for el in self.centroids]
                minimum = np.argmin(distance)

                clusters.append(minimum)
                toCentroid[minimum].append(x[i])

            newCentroids = []
            for el in toCentroid:
                toAverage = np.array(el)
                newCentroids.append(np.array((np.average(toAverage[:, 0]), np.average(toAverage[:, 1]))))

            self.centroids = np.array(newCentroids)
            self.labels = clusters



        return clusters

    def euclidean_function(self, x, y):
        return np.sum((x-y)**2)**0.5

#Part 6 - plotting every step iteration
    def plotCluster(self, x):

        fig, ax = plt.subplots()

        ax.scatter(x[:, 0], x[:, 1], c=self.labels)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], color='r', marker='*')
        plt.show()

#Exercise 2 - computing silhouette
def silhoutte_samples(x, labels):
    pass

def euclideanDistance(x, y):
    pass


def plotClusters(df):

    df['class'] = kindOfCluster
    fig, ax = plt.subplots()
    ax.scatter(df.loc[:, 'x'], df.loc[:, 'y'], c=df.loc[:, 'class'])
    plt.show()

df = pd.read_csv('gauss.txt')

points = np.array([(df.loc[i, 'x'], df.loc[i, 'y']) for i in range(len(df.index))])
n_clusters = 10
#print(points[0])
#print(points[1])
#print(points[:, 0])


#cluster = Kmeans(n_clusters=n_clusters)

#kindOfCluster = cluster.fit_predict(points)

#plotClusters(df)

#Part 4
#Load chameleon dataset
df_chameleon = pd.read_csv("chameleon.txt")
points = np.array([(df_chameleon.loc[i, 'x'], df_chameleon.loc[i, 'y']) for i in range(len(df_chameleon.index))])
n_clusters = 20

cluster = Kmeans(n_clusters=n_clusters, max_iter=30)
kindOfCluster = cluster.fit_predict(points)
df_chameleon['class'] = kindOfCluster

plotClusters(df_chameleon)




print("Finito..")
