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

from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from functools import partial

def rmsle(y_true, y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def train(target, X_train, y_train, xgb_params={}):	
	model = xgb.XGBRegressor(**xgb_params)
	#model = xgb.XGBRegressor(seed=2017)
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	print("RMSLE for " + target + ": " + str(rmsle(y_train, y_train_pred)))
	return model

def save():
	X_test = pd.read_csv("test.csv").drop(['id'], axis=1) 
	out = pd.DataFrame()
	out['id'] = range(1, 601)
	out['formation_energy_ev_natom'] = model_f.predict(X_test)
	out['bandgap_energy_ev'] = model_b.predict(X_test)
	out.to_csv('test_predicion.csv', index=False)

X_train = pd.read_csv("train.csv")
formation_energy_ev_natom_train = X_train['formation_energy_ev_natom']
bandgap_energy_ev_train = X_train['bandgap_energy_ev']
X_train = X_train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)

def objective(space):
	xgb_params = {
        'max_depth': int(space['max_depth']),
        'colsample_bytree': space['colsample_bytree'],
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'seed': int(space['seed']),
        'min_child_weight': int(space['min_child_weight']),
        'reg_alpha': space['reg_alpha'],
        'reg_lambda': space['reg_lambda'],
        'n_estimators': 100
    }

	model = xgb.XGBRegressor(**xgb_params)
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	score = rmsle(y_train, y_train_pred)

	print("Score: {0}".format(score))    
	return{'loss':score, 'status': STATUS_OK }

def grid_search(y_train):
	space = {
	    'max_depth': hp.quniform ('x_max_depth', 5, 20, 1),
	    'colsample_bytree': hp.uniform ('x_colsample_bytree', 0.8, 1.),
	    'learning_rate': hp.uniform ('x_learning_rate', 0.05, 0.2),
	    'subsample': hp.uniform ('x_subsample', 0.7, 1.),
	    'seed': hp.quniform ('x_seed', 0, 10000, 50),
	    'min_child_weight': hp.quniform ('x_min_child_weight', 1, 10, 1),
	    'reg_alpha': hp.loguniform ('x_reg_alpha', 0., 1.),
	    'reg_lambda': hp.uniform ('x_reg_lambda', 0.7, 1.),
	}

	trials = Trials()
	best_params = fmin(fn=objective,
			           space=space,
			           algo=partial(tpe.suggest, n_startup_jobs=1),
			           max_evals=20,
			           trials=trials)

	print("The best params: ", best_params)

xgb_params_formation_en = {'colsample_bytree': 0.8868855402664466, 
						  'learning_rate': 0.16091647200277445, 
						  'max_depth': 15,
						  'min_child_weight': 7.0,
						  'reg_alpha': 1.0165635383300418,
						  'reg_lambda': 0.7547335926442656,
						  'seed': 1650,
						  'subsample': 0.7967172584287566}		

xgb_params_bandgap = {'colsample_bytree': 0.9237770813851746,
					 'learning_rate': 0.1987950929940091,
					 'max_depth': 14,
					 'min_child_weight': 3.0,
					 'reg_alpha': 1.0006971259085735,
					 'reg_lambda': 0.970213199907543,
					 'seed': 9950,
					 'subsample': 0.9686596047107787}

model_f = train('formation_energy_ev_natom', X_train, formation_energy_ev_natom_train, xgb_params_formation_en)
model_b = train('bandgap_energy_ev', X_train, bandgap_energy_ev_train, xgb_params_bandgap)
save()


#grid_search(formation_energy_ev_natom_train)
#grid_search(bandgap_energy_ev_train)

#formation_energy_ev_natom_train best params:  {'x_colsample_bytree': 0.8868855402664466, 'x_learning_rate': 0.16091647200277445, 'x_max_depth': 15.0, 'x_min_child_weight': 7.0, 'x_reg_alpha': 1.0165635383300418, 'x_reg_lambda': 0.7547335926442656, 'x_seed': 1650.0, 'x_subsample': 0.7967172584287566}
#bandgap_energy_ev_train best params:  {'x_colsample_bytree': 0.9237770813851746, 'x_learning_rate': 0.1987950929940091, 'x_max_depth': 14.0, 'x_min_child_weight': 3.0, 'x_reg_alpha': 1.0006971259085735, 'x_reg_lambda': 0.970213199907543, 'x_seed': 9950.0, 'x_subsample': 0.9686596047107787}