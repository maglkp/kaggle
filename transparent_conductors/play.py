import pandas as pd
import numpy as np
seed = 2018
np.random.seed(seed)

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import xgboost as xgb

def rmsle(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


X_train = pd.read_csv("train.csv")
y_train = X_train['formation_energy_ev_natom']
X_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)

model = xgb.XGBRegressor(seed=2017)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
rmsle(y_train, y_train_pred)
