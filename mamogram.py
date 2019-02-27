# %%
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import tree, ensemble, svm, neighbors
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve

# %%
# 7. Attribute Information:
#   1. BI-RADS assessment: 1 to 5 (ordinal)
#   2. Age: patient's age in years (integer)
#   3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#   4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#   5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#   6. Severity: benign=0 or malignant=1 (binominal)

df = pd.read_csv(
    './datasets/mammographic_masses.data.txt',
    delimiter=',',
    header=None,
    na_values=['?']
)
df.columns = ['bi', 'age', 'shape', 'margin', 'density', 'severity']
df.head()

# %%
len(df)
# %%
df.dropna(inplace=True)
df

# %%
# Data sep.
features = list(df.columns[:5])
y = df['severity']

# %%
df.describe()

# %%
scaler = preprocessing.StandardScaler()
scaled_X = scaler.fit_transform(df[features])
scaled_X


# %%
for column in features:
    feature_i = df[column]
    # feature_i.value_counts().plot(kind='bar')

    # norm dist
    feature_i.mean()
    mu, std = norm.fit(feature_i)

    plt.hist(feature_i, bins=25, density=True,
             edgecolor='black', color='green')

    title = '{} -> fit: mu={:0.4} std={:0.4}'.format(column, mu, std)
    plt.title(title)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()


# %%
# Outliers

for feature in features:
    df[feature].mean()
    std = feature_i.std()

    plt.boxplot(df[feature])
    plt.title(feature)
    plt.show()


# %%
# Decision tree
print(scaled_X.shape)
print(y.shape)

d_tree = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(
    scaled_X,
    y,
    test_size=0.2,
    random_state=0
)

d_tree.fit(X_train, y_train)

print(d_tree.score(X_test, y_test))

score = cross_val_score(d_tree, scaled_X, y, cv=10)

print(score)
print(score.mean())

d_tree.predict([scaled_X[0]])

# %%
# Display model
dot_data = StringIO()
tree.export_graphviz(
    d_tree,
    out_file=dot_data,
    feature_names=features
)

(graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())

# %%
predictions = d_tree.predict(X_test)
confusion_matrix(y_test, predictions)

# %%
fpr, tpr, _ = roc_curve(y_test, predictions)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.plot(fpr, tpr, color='orange')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])

# %%
# Random forest
regr = ensemble.RandomForestClassifier(n_estimators=100)

regr.fit(X_train, y_train)

score = cross_val_score(regr, scaled_X, y, cv=10)
score.mean()

# %% SVM
model = svm.SVC(kernel='linear', C=1.0)

score = cross_val_score(model, scaled_X, y, cv=10)
score.mean()


# %%
# KNN
model = neighbors.KNeighborsClassifier(n_neighbors=2)
score = cross_val_score(model, scaled_X, y, cv=10)
score.mean()


# %%
# Naive bayes

# %%
# neural
