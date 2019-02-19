# %%
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

incomes = np.random.normal(27000, 15000, 10000)
np.mean(incomes)

# %%
plt.hist(incomes, 50)
plt.show()

# %%
np.median(incomes)

# %%
incomes = np.append(incomes, [1000000000])
np.median(incomes)

# %%
np.mean(incomes)

# %%
ages = np.random.randint(18, high=90, size=500)
ages

# %%
stats.mode(ages)
