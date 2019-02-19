# %% 
import os 
import io
import numpy as np
from pandas import DataFrame

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, _, filename in os.walk(path):
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

data = data.append(dataFrameFromDirectory('./datasets/DataScience-Python3/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('./datasets/DataScience-Python3/emails/ham', 'ham'))

# TODO: continue with file parsing to DF.
