# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

vals = np.random.normal(0, 0.5, 10000)
plt.grid(zorder=0)
plt.hist(vals, 50, edgecolor='b', zorder=2)
plt.show()

# %%
np.percentile(vals, 50)

# %%
np.percentile(vals, 90)

# %%
np.percentile(vals, 99)

# %%
sp.skew(vals)

# %%
sp.kurtosis(vals)
