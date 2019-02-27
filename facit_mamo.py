# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
from pydotplus import graph_from_dot_data
from sklearn.externals.six import StringIO
from IPython.display import Image
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy
from sklearn import preprocessing
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'datasets'))
    print(os.getcwd())
except:
    pass
# %% [markdown]
# # Final Project
#
# ## Predict whether a mammogram mass is benign or malignant
#
# We'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
#
# This data contains 961 instances of masses detected in mammograms, and contains the following attributes:
#
#
#    1. BI-RADS assessment: 1 to 5 (ordinal)
#    2. Age: patient's age in years (integer)
#    3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
#    4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
#    5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
#    6. Severity: benign=0 or malignant=1 (binominal)
#
# BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.
#
# Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.
#
# A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.
#
# ## Your assignment
#
# Apply several different supervised machine learning techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=10). Apply:
#
# * Decision tree
# * Random forest
# * KNN
# * Naive Bayes
# * SVM
# * Logistic Regression
# * And, as a bonus challenge, a neural network using Keras.
#
# The data needs to be cleaned; many rows contain missing data, and there may be erroneous data identifiable as outliers as well.
#
# Remember some techniques such as SVM also require the input data to be normalized first.
#
# Many techniques also have "hyperparameters" that need to be tuned. Once you identify a promising approach, see if you can make it even better by tuning its hyperparameters.
#
# I was able to achieve over 80% accuracy - can you beat that?
#
# %% [markdown]
# ## Let's begin: prepare your data
#
# Start by importing the mammographic_masses.data.txt file into a Pandas dataframe (hint: use read_csv) and take a look at it.

# %%
import pandas as pd

masses_data = pd.read_csv(
    '/Users/noone/Documents/companies/roxell/ml/datasets/mammographic_masses.data.txt')
masses_data.head()

# %% [markdown]
# Make sure you use the optional parmaters in read_csv to convert missing data (indicated by a ?) into NaN, and to add the appropriate column names (BI_RADS, age, shape, margin, density, and severity):

# %%
masses_data = pd.read_csv('/Users/noone/Documents/companies/roxell/ml/datasets/mammographic_masses.data.txt.data.txt', na_values=['?'], names=[
                          'BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
masses_data.head()

# %% [markdown]
# Evaluate whether the data needs cleaning; your model is only as good as the data it's given. Hint: use describe() on the dataframe.

# %%
masses_data.describe()

# %% [markdown]
# There are quite a few missing values in the data set. Before we just drop every row that's missing data, let's make sure we don't bias our data in doing so. Does there appear to be any sort of correlation to what sort of data has missing fields? If there were, we'd have to try and go back and fill that data in.

# %%
masses_data.loc[(masses_data['age'].isnull()) |
                (masses_data['shape'].isnull()) |
                (masses_data['margin'].isnull()) |
                (masses_data['density'].isnull())]

# %% [markdown]
# If the missing data seems randomly distributed, go ahead and drop rows with missing data. Hint: use dropna().

# %%
masses_data.dropna(inplace=True)
masses_data.describe()

# %% [markdown]
# Next you'll need to convert the Pandas dataframes into numpy arrays that can be used by scikit_learn. Create an array that extracts only the feature data we want to work with (age, shape, margin, and density) and another array that contains the classes (severity). You'll also need an array of the feature name labels.

# %%
all_features = masses_data[['age', 'shape',
                            'margin', 'density']].values


all_classes = masses_data['severity'].values

feature_names = ['age', 'shape', 'margin', 'density']

all_features

# %% [markdown]
# Some of our models require the input data to be normalized, so go ahead and normalize the attribute data. Hint: use preprocessing.StandardScaler().

# %%

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
all_features_scaled

# %% [markdown]
# ## Decision Trees
#
# Before moving to K-Fold cross validation and random forests, start by creating a single train/test split of our data. Set aside 75% for training, and 25% for testing.

# %%

numpy.random.seed(1234)

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)

# %% [markdown]
# Now create a DecisionTreeClassifier and fit it to your training data.

# %%

clf = DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set
clf.fit(training_inputs, training_classes)

# %% [markdown]
# Display the resulting decision tree.

# %%

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=feature_names)
graph = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# %% [markdown]
# Measure the accuracy of the resulting decision tree model using your test data.

# %%
clf.score(testing_inputs, testing_classes)

# %% [markdown]
# Now instead of a single train/test split, use K-Fold cross validation to get a better measure of your model's accuracy (K=10). Hint: use model_selection.cross_val_score

# %%

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()

# %% [markdown]
# Now try a RandomForestClassifier instead. Does it perform better?

# %%

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()

# %% [markdown]
# ## SVM
#
# Next try using svm.SVC with a linear kernel. How does it compare to the decision tree?

# %%

C = 1.0
svc = svm.SVC(kernel='linear', C=C)


# %%
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)

cv_scores.mean()

# %% [markdown]
# ## KNN
# How about K-Nearest-Neighbors? Hint: use neighbors.KNeighborsClassifier - it's a lot easier than implementing KNN from scratch like we did earlier in the course. Start with a K of 10. K is an example of a hyperparameter - a parameter on the model itself which may need to be tuned for best results on your particular data set.

# %%

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()

# %% [markdown]
# Choosing K is tricky, so we can't discard KNN until we've tried different values of K. Write a for loop to run KNN with K values ranging from 1 to 50 and see if K makes a substantial difference. Make a note of the best performance you could get out of KNN.

# %%
for n in range(1, 50):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print(n, cv_scores.mean())

# %% [markdown]
# ## Naive Bayes
#
# Now try naive_bayes.MultinomialNB. How does its accuracy stack up?

# %%

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(all_features)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, all_classes, cv=10)

cv_scores.mean()

# %% [markdown]
# ## Revisiting SVM
#
# svm.SVC may perform differently with different kernels. The choice of kernel is an example of a "hyperparamter." Try the rbf, sigmoid, and poly kernels and see what the best-performing kernel is. Do we have a new winner?

# %%
C = 1.0
svc = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# %%
C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()


# %%
C = 1.0
svc = svm.SVC(kernel='poly', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()

# %% [markdown]
# ## Logistic Regression
#
# We've tried all these fancy techniques, but fundamentally this is just a binary classification problem. Try Logisitic Regression, which is a simple way to tackling this sort of thing.

# %%

clf = LogisticRegression()
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
cv_scores.mean()

# %% [markdown]
# ## Neural Networks
#
# As a bonus challenge, let's see if an artificial neural network can do even better. You can use Keras to set up a neural network with 1 binary output neuron and see how it performs. Don't be afraid to run a large number of epochs to train the model if necessary.

# %%


def create_model():
    model = Sequential()
    # 4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model; adam seemed to work best
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# %%

# Wrap our Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features_scaled, all_classes, cv=10)
cv_scores.mean()

# %% [markdown]
# ## Do we have a winner?
#
# Which model, and which choice of hyperparameters, performed the best? Feel free to share your results!
# %% [markdown]
# ### The only clear loser is decision trees! Every other algorithm could be tuned to produce comparable results with 79-80% accuracy.
#
# Additional hyperparameter tuning, or different topologies of the multi-level perceptron might make a difference.

# %%
