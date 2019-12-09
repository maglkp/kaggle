#!/home/lkoziol/anaconda3/bin/python
import pandas as pd
import numpy as np

import os
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.vgg16 import VGG16
from keras.applications import DenseNet121
from keras.optimizers import Adam

from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2


seed = 2020
imgSize = 200 #500
numChannels = 3
numClasses = 5
train_dir = "train"
test_dir = "test"

# read
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

def read_data(df, df_dir):
    features = []
    target = []
    images = df['id_code'].values
    for img_id in images:
        image_file = df_dir + "/" + img_id + '.png'
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgSize, imgSize))
        features.append(image)    

    features = np.asarray(features)
    #features = features.astype('float32')
    #features /= 255    
    return features

X_train, y_train = read_data(train_df, train_dir), np.asarray(train_df.diagnosis)
y_trainc = np_utils.to_categorical(y_train)

input_shape = (imgSize, imgSize, numChannels)


# transform


# train
densenet = DenseNet121(
    include_top=False,
    input_shape=input_shape
)

model = Sequential([
    densenet,    
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(numClasses, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.00005),
    metrics=['accuracy']
)

model.fit(X_train, y_trainc, batch_size=100, epochs=2)
