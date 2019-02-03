#!/home/lkoziol/anaconda3/bin/python

# some useful links
# https://www.kaggle.com/nitinaggarwal008/eda-step-by-step-guide

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


df = pd.read_csv('train_V2.csv').dropna()
df_test = pd.read_csv('test_V2.csv')
df.head()

#df2 = df[0:50000]

cols_all = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'matchDuration', 'maxPlace',
       'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills',
       'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance',
       'weaponsAcquired', 'winPoints']
cols = ['assists', 'boosts']       
X_train = df[cols_all]
y_train = df['winPlacePerc']

model = xgb.XGBRegressor(seed=2019)
#model = RandomForestRegressor()
model.fit(X_train, y_train)

# 200 - 20s
# 1k - 40s
# 5k - 4m
# 8k - 8m
# 50k - 102m

# 5 features, 20k - 20m