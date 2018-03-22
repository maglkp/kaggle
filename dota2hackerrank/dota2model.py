#!/Users/koziol/anaconda3/bin/python3

import pandas as pd
import numpy as np

seed = 2018
np.random.seed(seed)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score

import xgboost as xgb

from hyperopt import hp

TRAIN_FILE = 'trainingdata.txt'
TRAIN_LEN = 15000
HEROES = ['Disruptor', 'Batrider', 'Naga Siren', 'Brewmaster', 'Lone Druid', 'Enigma', 'Windrunner',
          'Drow Ranger', 'Zeus', 'Earthshaker', 'Queen of Pain', 'Axe', 'Necrolyte', 'Lina', 'Phantom Assassin',
          'Vengeful Spirit', 'Witch Doctor', 'Alchemist', 'Faceless Void', 'Sniper', 'Puck', 'Tidehunter', 'Lich',
          'Night Stalker', 'Crystal Maiden', 'Outworld Devourer', 'Bounty Hunter', 'Death Prophet', 'Slardar',
          'Lifestealer', 'Pugna', 'Templar Assassin', 'Ogre Magi', 'Enchantress', 'Ancient Apparition', 'Pudge',
          'Timbersaw', 'Spirit Breaker', 'Chaos Knight', 'Visage', 'Bloodseeker', 'Shadow Demon', 'Meepo',
          'Nyx Assassin', 'Rubick', 'Undying', 'Kunkka', 'Sven', 'Troll Warlord', 'Leshrac', 'Chen', 'Magnus', 'Huskar',
          'Slark', 'Bane', 'Morphling', 'Centaur Warrunner', 'Broodmother', 'Storm Spirit', 'Viper', 'Skeleton King',
          'Clinkz', 'Tiny', 'Anti-Mage', 'Lion', 'Tinker', 'Phantom Lancer', 'Treant Protector', 'Lycanthrope',
          'Gyrocopter', 'Mirana', 'Luna', 'Beastmaster', 'Shadow Shaman', 'Spectre', 'Invoker', 'Keeper of the Light',
          'Omniknight', 'Shadow Fiend', 'Dark Seer', 'Weaver', 'Warlock', 'Jakiro', 'Dragon Knight', 'Ursa',
          'Juggernaut', 'Razor', 'Clockwerk', 'Wisp', 'Doom', 'Silencer', 'Riki', 'Sand King', 'Medusa',
          "Nature's Prophet", 'Dazzle', 'Venomancer']


def get_hero_names():
    heroes = set()
    with open(TRAIN_FILE, 'r') as f:
        for line in f:
            for hero in line.split(',')[:-1]:
                heroes.add(hero)
    return heroes


def get_column_names():
    t1 = [h + '1' for h in HEROES]
    t2 = [h + '2' for h in HEROES]
    return t1 + t2 + ['team1win']


def read_train_df():
    train = pd.DataFrame(0, index=np.arange(TRAIN_LEN), columns=get_column_names())
    # Sven,Lone Druid,Venomancer,Clockwerk,Shadow Shaman,Invoker,Gyrocopter,Anti-Mage,Alchemist,Slark,2
    with open(TRAIN_FILE, 'r') as f:
        i = 0
        for line in f:
            dat = line.split(',')
            team1 = [m + '1' for m in dat[0:5]]
            team2 = [m + '2' for m in dat[5:10]]
            members = team1 + team2
            team1win = 1 if dat[-1] == '1' else 0
            row = train.iloc[i]
            for member in members:
                row[member] = 1
            row['team1win'] = team1win
            i += 1
    return train


train = read_train_df()
