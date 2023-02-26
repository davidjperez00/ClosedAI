# util-FMNIST

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import matplotlib.pyplot as plt
import pickle
import numpy as np


# Load the FMNIST dataset
def loadFashionMNIST():
    # Load data from the fashion_mnist dataset
    (trainFullX, trainFullY), (testX, testY) = keras.datasets.fashion_mnist.load_data()
    
    # Normalize the data into the range [0, 1]
    trainFullX = trainFullX / 255.0
    testX = testX / 255.0
    
    # Split the data into separate training and validation chunks
    trainX = trainFullX[5000:]
    trainY = trainFullY[5000:]
    validX = trainFullX[:5000]
    validY = trainFullY[:5000]
    
    return (trainX, trainY), (testX, testY), (validX, validY)
    

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

# Print charts showing the accuracy and loss of the given model
def plot_model(loadedModel):
    # Accuracy
    training = plt.plot(loadedModel.get('accuracy'), 'b-', label = 'Training Accuracy')
    plt.plot(loadedModel.get('accuracy'), 'bo')
    validation = plt.plot(loadedModel.get('val_accuracy'), 'r-', label = 'Validation Accuracy')
    plt.plot(loadedModel.get('val_accuracy'), 'ro')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Loss
    training = plt.plot(loadedModel.get('loss'), 'b-', label = 'Training Loss')
    plt.plot(loadedModel.get('loss'), 'bo')
    validation = plt.plot(loadedModel.get('val_loss'), 'r-', label = 'Validation Loss')
    plt.plot(loadedModel.get('val_loss'), 'ro')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()