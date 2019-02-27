# %%

import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

# %%
input_file = './datasets/PastHires.csv'

df = pd.read_csv(input_file)

# %%

df.head()
df.columns
# %%
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)

sch = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(sch)

df.head()

# %%
# Extract features
features = list(df.columns[:6])
features

# %%
# Build the tree
y = df['Hired']
X = df[features]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# %%
# Display model
dot_data = StringIO()
tree.export_graphviz(
    clf,
    out_file=dot_data,
    feature_names=features
)

(graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())


# %%
