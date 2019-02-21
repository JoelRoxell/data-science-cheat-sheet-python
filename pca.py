# %%
# Principle component analysis

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle

iris = load_iris()

num_samples, num_features = iris.data.shape

print(num_samples, num_features)

print(list(iris.target_names))

# %%

X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)

X_pca = pca.transform(X)

# %%
print(pca.components_)

# How much of the previous variation is preserved
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

# %%
colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()

for i, c, label in zip(target_ids, colors, iris.target_names):
    pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1], c=c, label=label)

pl.legend()
pl.show()


