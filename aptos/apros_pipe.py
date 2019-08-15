#!/home/lkoziol/anaconda3/bin/python
import pandas as pd
import numpy as np

import os
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin

import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2

seed = 2020
imgSize = 200  # 500
numChannels = 3
numClasses = 5
train_dir = "train"
test_dir = "test"
input_shape = (imgSize, imgSize, numChannels)

# read
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# train_df = train_df[:500]

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
    return features


X_train, y_train = read_data(train_df, train_dir), np.asarray(train_df.diagnosis)
y_trainc = np_utils.to_categorical(y_train)


# transform
class DfTransform(TransformerMixin, BaseEstimator):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def fit(self, *_):
        return self

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)


# train
def densenetModel():
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
    return model


pipeline = Pipeline([
    ('model', densenetModel())
])

# pipeline.fit(X_train, y_trainc, model__batch_size=50, model__epochs=5)


model = densenetModel()
model.fit(X_train, y_trainc, batch_size=50, epochs=5)
