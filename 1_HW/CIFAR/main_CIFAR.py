# File Name: main_CIFAR.py
# Brief: Utility functions and data retrieval for CIFAR-100 nueral net.
# Date: 2/17/2023
# Authors: David Perez

import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils_CIFAR
import model_CIFAR

MODEL_ONE_SAVE_PATH = 'cifar100_model_1'
MODEL_ONE_HISTORY_SAVE_PATH = 'cifar100_model_1_history.pkl"'
MODEL_TWO_SAVE_PATH = 'cifar100_model_2'
MODEL_TWO_HISTORY_SAVE_PATH = 'cifar100_model_1_history.pkl"'

def main():
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # ---------- TRAINING OUR FIRST MODEL ----------- #
    # Load the CIFAR-100 dataset using a scaled dataset
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = utils_CIFAR.load_cifar100_dataset()
    
    cifar_model_one = model_CIFAR.create_cifar_model_one()
    
    cifar_model_one = model_CIFAR.compile_cifar_model_one(cifar_model_one)
    
    model_one_history = model_CIFAR.train_cifar_model_one(cifar_model_one,
                            X_train, y_train, X_valid, y_valid)
    
    # Save model training history to be used later for plotting training
    utils_CIFAR.save_pkl_model_history(model_one_history, MODEL_ONE_HISTORY_SAVE_PATH)
    
    # Save our model for later analysis
    utils_CIFAR.save_keras_model(cifar_model_one, MODEL_ONE_SAVE_PATH)
    
    # ---------- TRAINING OUR SECOND MODEL ----------- #
    # Load the CIFAR-100 dataset using a scaled dataset
    (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = utils_CIFAR.load_cifar100_dataset()
    
    cifar_model_two = model_CIFAR.create_cifar_model_two()
    
    cifar_model_two = model_CIFAR.compile_cifar_model_two(cifar_model_two)
    
    model_two_history = model_CIFAR.train_cifar_model_two(cifar_model_two,
                          X_train, y_train, X_valid, y_valid)
    
    # Save model training history to be used later for plotting training
    utils_CIFAR.save_pkl_model_history(model_two_history, MODEL_TWO_HISTORY_SAVE_PATH)
    
    # Save our model for later analysis
    utils_CIFAR.save_keras_model(cifar_model_two, MODEL_TWO_SAVE_PATH)

if __name__ == '__main__':
    main()

