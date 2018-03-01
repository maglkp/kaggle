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

def rmsle(y_true, y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def train(target, X_train, y_train):		
	model = xgb.XGBRegressor(seed=2017)
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	print("RMSLE for " + target + ": " + str(rmsle(y_train, y_train_pred)))
	return model

X_train = pd.read_csv("train.csv")
formation_energy_ev_natom_train = X_train['formation_energy_ev_natom']
bandgap_energy_ev_train = X_train['bandgap_energy_ev']
X_train = X_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)

model_f = train('formation_energy_ev_natom', X_train, formation_energy_ev_natom_train)
model_b = train('bandgap_energy_ev', X_train, bandgap_energy_ev_train)

out = pd.DataFrame()
X_test = pd.read_csv("test.csv").drop(['id'], axis=1) 
out['formation_energy_ev_natom'] = model_f.predict(X_test)
out['bandgap_energy_ev'] = model_b.predict(X_test)
out['id'] = range(1, 601)
out.to_csv('test_predicion.csv', index=False)