# %% matplotlib inline
import pandas as pd
import numpy as np

msg = "Hello Worlds"
print(msg)
df = pd.read_csv('./datasets/PastHires.csv')
df.head()

# %%
df.tail()

# %%
df.shape

# %%
df.columns

# %%
df['Hired']

# %%
df['Hired'][:5]

# %%
df[['Years Experience', 'Hired']]

# %%
df.sort_values(['Years Experience'], ascending=False)

# %%
degree_count = df['Level of Education'].value_counts()
degree_count

# %%
degree_count.plot(kind='bar')

# %%
rows = df[['Previous employers', 'Hired']][5:11]
rows.plot(kind='hist')

# %%
rows
