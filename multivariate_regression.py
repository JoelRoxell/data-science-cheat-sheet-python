# %% 
import numpy as np 
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import statsmodels.api as sm

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

df.head()

# %%
df.columns

# %% 
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print(X)

est = sm.OLS(y, X).fit()

est.summary()
