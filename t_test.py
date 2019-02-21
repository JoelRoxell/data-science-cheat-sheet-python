# %% 
import numpy as np 
from scipy import stats 

A = np.random.normal(25.0, 5.0, 10000)
B = np.random.normal(26.0, 5.0, 10000)

# Results in a "bad stat (-)" and a really low p, which indicates that there is a really small chance that the produced  result actually was a result of pure chance / variation.
stats.ttest_ind(A, B)

# %%
B = np.random.normal(25.0, 5.0, 10000)

stats.ttest_ind(A, B)

# %%

A = np.random.normal(25.0, 5.0, 100000)
B = np.random.normal(25.0, 5.0, 100000)

stats.ttest_ind(A, B)

# %%

A = np.random.normal(25.0, 5.0, 1000000)
B = np.random.normal(25.0, 5.0, 1000000)

stats.ttest_ind(A, B)

# %%
stats.ttest_ind(A, A)