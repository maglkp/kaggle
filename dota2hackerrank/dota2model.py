#!/Users/koziol/anaconda3/bin/python3

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

TRAIN_FILE = 'trainingdata.txt'

def get_hero_names():
	heroes = set()
	with open(TRAIN_FILE, 'r') as f:
		for line in f:
			for hero in line.split(',')[:-1]:
				heroes.add(hero)
	return heroes

heroes = ['Disruptor', 'Batrider', 'Naga Siren', 'Brewmaster', 'Lone Druid', 'Enigma', 'Windrunner', 
		  'Drow Ranger', 'Zeus', 'Earthshaker', 'Queen of Pain', 'Axe', 'Necrolyte', 'Lina', 'Phantom Assassin', 
		  'Vengeful Spirit', 'Witch Doctor', 'Alchemist', 'Faceless Void', 'Sniper', 'Puck', 'Tidehunter', 'Lich', 'Night Stalker', 'Crystal Maiden', 'Outworld Devourer', 'Bounty Hunter', 'Death Prophet', 'Slardar', 'Lifestealer', 'Pugna', 'Templar Assassin', 'Ogre Magi', 'Enchantress', 'Ancient Apparition', 'Pudge', 'Timbersaw', 'Spirit Breaker', 'Chaos Knight', 'Visage', 'Bloodseeker', 'Shadow Demon', 'Meepo', 'Nyx Assassin', 'Rubick', 'Undying', 'Kunkka', 'Sven', 'Troll Warlord', 'Leshrac', 'Chen', 'Magnus', 'Huskar', 'Slark', 'Bane', 'Morphling', 'Centaur Warrunner', 'Broodmother', 'Storm Spirit', 'Viper', 'Skeleton King', 'Clinkz', 'Tiny', 'Anti-Mage', 'Lion', 'Tinker', 'Phantom Lancer', 'Treant Protector', 'Lycanthrope', 'Gyrocopter', 'Mirana', 'Luna', 'Beastmaster', 'Shadow Shaman', 'Spectre', 'Invoker', 'Keeper of the Light', 'Omniknight', 'Shadow Fiend', 'Dark Seer', 'Weaver', 'Warlock', 'Jakiro', 'Dragon Knight', 'Ursa', 'Juggernaut', 'Razor', 'Clockwerk', 'Wisp', 'Doom', 'Silencer', 'Riki', 'Sand King', 'Medusa', "Nature's Prophet", 'Dazzle', 'Venomancer']
