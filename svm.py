# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
def createClusteredData(N, k):

    pointsPerCluster = float(N) / k

    X = []
    y = []

    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)

        for j in range(int(pointsPerCluster)):
            X.append([
                np.random.normal(incomeCentroid, 10000.0),
                np.random.normal(ageCentroid, 2.0)
            ])

            y.append(i)
            

    X = np.array(X)
    y = np.array(y)

    return X, y


# %%
(X, y) = createClusteredData(100, 5)

plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))

# %%
X

# %%
y

# %%
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)

def plot(clf):
    xx, YY = np.meshgrid(np.arange(0, 250000, 10), np.arange(10, 70, 0.5))

    Z = clf.predict(np.c_[xx.ravel(), YY.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, YY, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

plot(svc)

