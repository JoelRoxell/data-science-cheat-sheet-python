# %%
import numpy as np
from matplotlib.pyplot import scatter

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

scatter(pageSpeeds, purchaseAmount, edgecolors='b')

# %%
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

scatter(pageSpeeds, purchaseAmount, edgecolors='b')

# %%
np.corrcoef(pageSpeeds, purchaseAmount)

# %%
np.cov(pageSpeeds, purchaseAmount)
