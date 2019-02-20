# %%

import sklearn
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt 

def createClusteredData(N, k):
    np.random.seed(10)

    pointsPerCluster = float(N) / k
    X = []

    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)

        for j in range(int(pointsPerCluster)):
            X.append([
                np.random.normal(incomeCentroid, 10000.0),
                np.random.normal(ageCentroid, 2.0)
            ])

    X = np.array(X)

    return X


# %%

data = createClusteredData(100, 5)

model = KMeans(n_clusters=4)

# use scale to normalize the input
model = model.fit(sklearn.preprocessing.scale(data))

print(model.labels_)

plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float))
plt.show()
