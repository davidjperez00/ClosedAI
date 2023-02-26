# main-FMNIST

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt
import pickle
import numpy as np
import utils_FMNIST
import model_FMNIST

MODEL_TWO_SAVE_PATH_REG = 'mnist_model_2_reg'
MODEL_TWO_HISTORY_SAVE_PATH_REG = 'mnist_model_2_history_reg.pkl'

MODEL_TWO_SAVE_PATH = 'mnist_model_2'
MODEL_TWO_HISTORY_SAVE_PATH = 'mnist_model_2_history.pkl'

MODEL_TWO_SAVE_PATH_REG_ALT = 'mnist_model_2_reg_alt'
MODEL_TWO_HISTORY_SAVE_PATH_REG_ALT = 'mnist_model_2_history_reg_alt.pkl'

MODEL_TWO_SAVE_PATH_ALT = 'mnist_model_2_alt'
MODEL_TWO_HISTORY_SAVE_PATH_ALT = 'mnist_model_2_history_alt.pkl'


def main(): 
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # ---------- TRAINING OUR FIRST MODEL ----------- #



    # ---------- TRAINING OUR SECOND MODEL ----------- #
    (trainX, trainY), (testX, testY), (validX, validY) = utils_FMNIST.loadFashionMNIST()

    # Second model without regularizers, first hyperparameter set
    fashionModelTwo = model_FMNIST.createModelTwoMNIST(False)
    fashionModelTwo = model_FMNIST.compileModelTwoMNIST(fashionModelTwo, 5e-4)
    modelTwoHistory = model_FMNIST.trainModelTwoMNIST(fashionModelTwo, trainX, trainY, validX, validY, 50)

    utils_FMNIST.save_pkl_model_history(modelTwoHistory, MODEL_TWO_HISTORY_SAVE_PATH)
    utils_FMNIST.save_keras_model(fashionModelTwo, MODEL_TWO_SAVE_PATH)

    # Second model with regularizers, first hyperparameter set
    fashionModelTwo = model_FMNIST.createModelTwoMNIST(True)
    fashionModelTwo = model_FMNIST.compileModelTwoMNIST(fashionModelTwo, 5e-4)
    modelTwoHistory = model_FMNIST.trainModelTwoMNIST(fashionModelTwo, trainX, trainY, validX, validY, 50)

    utils_FMNIST.save_pkl_model_history(modelTwoHistory, MODEL_TWO_HISTORY_SAVE_PATH_REG)
    utils_FMNIST.save_keras_model(fashionModelTwo, MODEL_TWO_SAVE_PATH_REG)
    
    # Second model without regularizers, second hyperparameter set
    fashionModelTwo = model_FMNIST.createModelTwoMNIST(False)
    fashionModelTwo = model_FMNIST.compileModelTwoMNIST(fashionModelTwo, 5e-3)
    modelTwoHistory = model_FMNIST.trainModelTwoMNIST(fashionModelTwo, trainX, trainY, validX, validY, 25)

    utils_FMNIST.save_pkl_model_history(modelTwoHistory, MODEL_TWO_HISTORY_SAVE_PATH)
    utils_FMNIST.save_keras_model(fashionModelTwo, MODEL_TWO_SAVE_PATH)

    # Second model with regularizers, second hyperparameter set
    fashionModelTwo = model_FMNIST.createModelTwoMNIST(True)
    fashionModelTwo = model_FMNIST.compileModelTwoMNIST(fashionModelTwo, 5e-3)
    modelTwoHistory = model_FMNIST.trainModelTwoMNIST(fashionModelTwo, trainX, trainY, validX, validY, 25)

    utils_FMNIST.save_pkl_model_history(modelTwoHistory, MODEL_TWO_HISTORY_SAVE_PATH_REG)
    utils_FMNIST.save_keras_model(fashionModelTwo, MODEL_TWO_SAVE_PATH_REG)

main()