# model-FMNIST

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils_FMNIST import *


# Create the architecture for FMNIST model 1
def createModelOneMNIST():
    return 0

# Create the architecture for FMNIST model 2
def createModelTwoMNIST(useRegularizers):
    modelWith = keras.models.Sequential([
        keras.layers.Dense(units=64, input_shape=[28, 28], activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=128, activity_regularizer = regularizers.L2(1e-5)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=128, activity_regularizer = regularizers.L2(1e-5)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=256, activity_regularizer = regularizers.L2(1e-5)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=256, activity_regularizer = regularizers.L2(1e-5)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activity_regularizer = regularizers.L2(1e-5)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=100, activation='softmax'),
    ])
    
    modelWithout = keras.models.Sequential([
        keras.layers.Dense(units=64, input_shape=[28, 28], activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(units=256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=100, activation='softmax'),
    ])
  
    if useRegularizers:
        return modelWith
    else:
        return modelWithout


# Compile model 2 with the given learning rate
def compileModelTwoMNIST(model, learningRate):
    optimizer = keras.optimizers.Nadam(learning_rate=learningRate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics=["accuracy"])

    return model


# Train model 2 with the given number of maximum epochs
def trainModelTwoMNIST(model, X_train, y_train, X_valid, y_valid, _epochs):
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)

    history = model.fit(X_train, y_train, epochs=_epochs, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb])

    return history