# %%
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

# %% 
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

clf.score(X_test, Y_test)

# %%
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

print(scores)

print(scores.mean())

# %%
print(clf.predict([iris.data[0]]))
print(iris.target[0])
