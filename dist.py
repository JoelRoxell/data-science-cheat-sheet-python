# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, binom, poisson

values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, 50, edgecolor='black')
plt.show()

# %%
# Normal / Gaussian
x = np.arange(-3, 3, 0.001)
plt.plot(x, norm.pdf(x))

# %%
mu = 5.0
sigma = 2.0
values = np.random.normal(mu, sigma, 10000)
plt.hist(values, 50, edgecolor='r')

# %%
# exponential pdf

x = np.arange(0, 10, 0.001)
plt.plot(x, expon.pdf(x))

# %%
# Binomial prb. mass function

n, p = 10, 0.5
x = np.arange(0, 10, 0.001)
plt.plot(x, binom.pmf(x, n, p))

# %%
# poisson probability mass func
mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))
