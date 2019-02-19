# %%
# Matplots basics

# Line graph
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

x = np.arange(-3, 3, 0.001)

plt.plot(x, norm.pdf(x))
plt.show()

# %%
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# %%
# Save plt
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.savefig('./tmp.png', format='png')

# %%
# Adjust Axes
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.grid()
axes.set_xticks(np.arange(-5, 6, 1))

plt.xlabel('gg')
plt.ylabel('wp')

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')

plt.legend(['first', 'second'], loc=4)

plt.show()

# %%
# Bar chart
val = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
plt.bar(range(0, 5), val, color=colors)

# %%
# scatter
X = np.random.randn(500)
Y = np.random.randn(500)

plt.scatter(X, Y, edgecolors='black')
plt.show()

# %%
# hist
incomes = np.random.normal(27000, 15000, 10000)
plt.hist(incomes, 20, edgecolor='black')
plt.show()

# %%
# box
uni_skew = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100
data = np.concatenate((uni_skew, high_outliers, low_outliers))

plt.boxplot(data)
plt.show()
