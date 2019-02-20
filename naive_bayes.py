# %% 
import os 
import io
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def readFiles(path):
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            lines = []
            inBody = False

            with io.open(file_path, 'r', encoding='latin1') as file:
                for line in file:
                    if inBody:
                        lines.append(line)
                    elif line == '\n':
                        inBody = True

            message = '\n'.join(lines)

            yield path, message

def dataFrameFromDirectory(path, classification):      
    rows = []
    index = []

    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('./data-science-cheat-sheet-python/datasets/DataScience-Python3/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('./data-science-cheat-sheet-python/datasets/DataScience-Python3/emails/ham', 'ham'))

# %%
data.head()

# %% 
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
labels = data['class'].values

classifier.fit(counts, labels)

# %% 
examples = ['Free Viagra now!!!', 'Hi m8, how about a game of fotball tmr?']
example_counts = vectorizer.transform(examples)

predictions = classifier.predict(example_counts)
predictions

# %%
# split up in train/test
train, test = train_test_split(data, test_size=0.2)

classifier = MultinomialNB()

counts = vectorizer.fit_transform(train['message'].values)
labels = train['class'].values

classifier.fit(counts, labels)

counts_test = vectorizer.transform(test['message'].values)

predictions = classifier.predict(counts_test)

predictions

# %%

print(confusion_matrix(test['class'].values, predictions))
print(accuracy_score(test['class'].values, predictions))
