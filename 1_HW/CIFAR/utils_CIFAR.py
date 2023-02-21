# File Name: utils_CIFAR.py
# Brief: Utility functions and data retrieval for CIFAR-100 nueral net.
# Date: 2/17/2023
# Authors: David Perez

import tensorflow as tf
from tensorflow import keras
import pickle

# @brief Load CIFAR-100 into train, test, and validation sets.
#
# @note CIFAR-100 is a dataset of 32x32 pixel 3 channel RGB images. 
#     The there are 100 classes containing 600 images each. Each label contains
#     a general superclass as well as a subclass denoting the actual objects name.
def load_cifar100_dataset():
  (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar100.load_data()

  X_train = X_train_full[5000:]
  y_train = y_train_full[5000:] 
  X_valid = X_train_full[:5000] 
  y_valid = y_train_full[:5000] 
  
  # Scale RGB values to be 0-1
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_valid = X_valid.astype('float32')

  X_train = X_train / 255.
  X_test = X_test / 255.
  X_valid = X_valid / 255.

  return (X_train, y_train), (X_test, y_test), (X_valid, y_valid)

# @brief Save model training history in dictionary format using `pickle`.
#
# @param[in] history History returned when calling fit method on model.
# @param[in] pkl_file_name A file_name.pkl string to save model training history
def save_pkl_model_history(history, pkl_file_name):
  with open(pkl_file_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# @brief Load dictionary of model data
#
# @param[in] pkl_file_name A .pkl file containing model history
#     EX: 'model_one_history.pkl'
#
# @return Returns a dictionary containing 'loss', 'val_loss', 
#     'accuracy', and 'val_accuracy'.
def load_pkl_model_history(pkl_file_name):
  with open(pkl_file_name, "rb") as file_pi:
    history = pickle.load(file_pi)

  return history

# @brief Function for saving keras model with weights
#
# @param[in] model Keras model to be saved
# @param[in] save_path a string denoting save path (Creates folder with this name)
def save_keras_model(model, save_path):
  tf.keras.Model.save(model, save_path)

# @brief Function for loading a saved keras model
#
# @param[in] model_name A string containing the location of model folder name
def load_keras_model(model_name):
  model = tf.keras.models.load_model(model_name)

  return model