# File Name: utils-CIFAR.py
# Brief: Utility functions and data retrieval for CIFAR-100 nueral net.
# Date: 2/17/2023
# Authors: David Perez

import tensorflow as tf
from tensorflow import keras
# from utils_CIFAR import *

def create_cifar_model_one():
  model = keras.models.Sequential([
      keras.layers.Conv2D(filters=64, kernel_size=7, input_shape=[32, 32, 3], activation="relu", padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Conv2D(filters=256, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Flatten(),
      keras.layers.Dense(units=128),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(units=100, activation='softmax'),
  ])

  return model

def create_cifar_model_two():
  model = keras.models.Sequential([
      keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=[32, 32, 3], activation="relu", padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.Conv2D(filters=64, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Dropout(0.1),
      keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.Conv2D(filters=128, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Dropout(0.2),
      keras.layers.Conv2D(filters=256, kernel_size=3, padding="SAME"),
      keras.layers.Activation("relu"),
      keras.layers.Conv2D(filters=256, kernel_size=3, padding="SAME"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.MaxPooling2D(pool_size=2),
      keras.layers.Flatten(),
      keras.layers.Dense(units=512),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(units=100, activation='softmax'),
  ])
  
  return model

def compile_cifar_model_one(model):
  optimizer = keras.optimizers.Adam(learning_rate=5e-3)
  model.compile(loss="sparse_categorical_crossentropy",
                optimizer=optimizer, metrics=["accuracy"])

  return model

def compile_cifar_model_two(model):
  optimizer = keras.optimizers.Nadam(learning_rate=5e-4)
  model.compile(loss="sparse_categorical_crossentropy",
                optimizer= optimizer, metrics=["accuracy"])

  return model

def train_cifar_model_one(model, X_train, y_train, X_valid, y_valid):
  early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)

  history = model.fit(X_train, y_train, epochs=20, 
    validation_data=(X_valid, y_valid),  callbacks=[early_stopping_cb])

  return history

def train_cifar_model_two(model, X_train, y_train, X_valid, y_valid):
  early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)

  history = model.fit(X_train, y_train, epochs=50, 
    validation_data=(X_valid, y_valid),  callbacks=[early_stopping_cb])

  return history
