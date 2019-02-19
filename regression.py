# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

pagespeeds = np.random.normal(3.0, 2.0, 1000)
purchase_amount = 100 - (pagespeeds + np.random.normal(0, 1, 1000)) * 3

plt.scatter(pagespeeds, purchase_amount, edgecolors='black')

# %% 
slope, intercept, r_value, p_value, std_err = stats.linregress(
    pagespeeds,
    purchase_amount
)

# %%
r_value ** 2

# %%

def predict(x):
    return slope * x + intercept

fitLine = predict(pagespeeds)

plt.scatter(pagespeeds, purchase_amount, edgecolors='black')
plt.plot(pagespeeds, fitLine, c='r')
plt.show()

# %%
np.random.seed(2)
pagespeeds = np.random.normal(3.0, 1.0, 1000)
purchase_amount = np.random.normal(50.0, 10.0, 1000) / pagespeeds

plt.scatter(pagespeeds, purchase_amount, edgecolors='b')
plt.show()

# %%
x = np.array(pagespeeds)
y = np.array(purchase_amount)
p4 = np.poly1d(np.polyfit(x, y, 5))

p4

# %% 
xp = np.linspace(0, 7, 100)
plt.scatter(x, y, edgecolors='black')
plt.plot(xp, p4(xp), c='r')
plt.show()

r2 = r2_score(y, p4(x))

print('r2: {:.4}'.format(r2))
