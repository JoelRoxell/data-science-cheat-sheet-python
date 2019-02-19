# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

np.random.seed(2)
page_speeds = np.random.normal(3.0, 1.0, 100)
purchase_amount = np.random.normal(50.0, 30.0, 100) / page_speeds

plt.scatter(page_speeds, purchase_amount)

# %%

# OBS: data should be shuffled if not randomized from start.
train_X = page_speeds[:80]
test_X = page_speeds[80:]

train_Y = purchase_amount[:80]
test_Y = purchase_amount[80:]

# %%
# Training set
plt.scatter(train_X, train_Y)

# %%
# Training set
plt.scatter(test_X, test_Y)

# %% 

x = np.array(train_X)
y = np.array(train_Y)

model = np.poly1d(np.polyfit(x, y, 5))

# %%
xp = np.linspace(0, 7, 80)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])

plt.scatter(x, y, edgecolors='black')
plt.plot(xp, model(xp), c='red')  

# %%
metrics.r2_score(test_Y, model(test_X))


# %%
metrics.r2_score(train_Y, model(train_X))


