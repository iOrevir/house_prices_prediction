# %%
# * 1. Prepare Problem
# & a) Load libraries
from os import SCHED_OTHER
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# %%
# & b) Load dataset
data = pd.read_csv("/Users/erivero/Documents/MBD_IE/t1/DATASCIENCE INDIVIDUAL COMPETITION 1ST ATTEMP MBD-EN2020S-2_32R569_381926/willow-real-estate/train.csv", header=0)

# %%
# * 2. Summarize Data
# %%
# & a) Descriptive statistics
# Descriptive statistics

# Peek at data
peek = data.head(20)
print(f"dataframe \n {peek}")

# Dimensions
shape = data.shape
print(f"shape: {shape}")

# Data types
types = data.dtypes
print(f"types: \n {types}")

# %%
# Statistical summary
pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = data.describe()
print(f"description; \n {description}")

# %%
# # Class distribution (classification only)
# class_counts = data.groupby(by='class').size()
# print(f"class distribution: \n {class_counts}")

# %%
# Pairwise Pearson correlation
correlations = data.select_dtypes(include='number').corr(method='pearson')
print(f"correlations: \n {correlations}")

# %%
# Skew for each attribute
skew = data.select_dtypes(include='number').skew()
print(f"skew: \n {skew}")

# %%
# & b) Data visualizations
# Histograms - distribution of each attribute
data.hist(figsize=(15,15))
plt.show()

# Density plots - distribution of each attribute
# distribution for each attribute is clearer than the histogram
data.plot(kind='density', subplots=True, layout=(8,3), sharex=False, figsize=(15,15))
plt.show()

# Box & whisker plots
data.plot(kind='box', subplots=True, layout=(8,3), sharex=False, figsize=(15,15))
plt.show()

# %%
# & pandas profiling

profile = ProfileReport(data, title = 'Pandas Profiling Report')

# save to file
profile.to_file('output.html')

"""  
Observations attributes:
- All but month and id show a high correlation
- coord_X & coord_Y have NaNs and city is constant
- high correlation with price (target): viewsToPOI, living_m2, baths, bearing & house_quality_index
"""
# %%
# * 3. Prepare Data
# & a) Data Cleaning
# find attributes with NaNs
data.isna().any()
# find unique values of some attributes
data.basement.unique()

# ! drop id, coord_X, coord_Y, city. Id is not needed. Coord_s have NaNs and they are highly correlated with other attributes. City is constant. Dow (day of the week) may not add to the prediction.


# ! dummyfy house_state_index, basement, viewsToPOI, view_quality.


# ! casting floors to numeric


# b) Feature Selection
# c) Data Transforms

# %%
# * 4. Evaluate Algorithms
# %%
# & a) Split-out validation dataset

# get values from df
array = data.values

# separate array into input & output
X = array[:, 0:23]       # input
Y = array[:, 23]           # output

# shares
validation_size = 0.20
seed = 13

# split
X_train, X_validation, Y_train, Y_validation= train_test_split(X, Y, test_size=validation_size, random_state=seed)

# %%
# & b) Test options and evaluation metric (RMSLE)
n_folds = 10
scoring = 'neg_mean_squared_log_error'

# %%
# & c) Spot Check Algorithms

# & baseline

# prepare models
models = []
models.append(('LR', LinearRegression())) 
models.append(('LASSO', Lasso()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('EN', ElasticNet()))
models.append(('SVR', SVR()))

# evaluate each model 
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean():f} ({cv_results.std():f})"
    print(msg)
    
# comparison boxplot
fig = plt.figure(figsize=(15,15))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# %%
# & d) Compare Algorithms

# %%
# * 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# %%
# * 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use