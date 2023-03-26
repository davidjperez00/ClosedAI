# import tensorflow_datasets as tfds
# from collections import Counter
import tensorflow as tf
from tensorflow import keras

def create_GRU_model(vocab_size, num_oov_buckets, embed_size = 128):
    inputs = keras.Input(shape=(None,))
    embedding = keras.layers.Embedding(vocab_size + num_oov_buckets, 
                                       embed_size, mask_zero = True, 
                                       input_shape = [None])(inputs)
    query = keras.layers.Conv1D(filters=100, kernel_size=4, padding="same")(embedding)
    value = keras.layers.Conv1D(filters=100, kernel_size=4, padding="same")(embedding)
    attention = keras.layers.Attention()([query, value])
    concat = keras.layers.Concatenate()([query, attention])
    gru1 = keras.layers.GRU(128, return_sequences = True)(concat)
    bn = keras.layers.BatchNormalization()(gru1)
    gru2 = keras.layers.GRU(128)(bn)
    outputs = keras.layers.Dense(1, activation = "sigmoid")(gru2)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_LSTM_model_1(vocab_size, num_oov_buckets, embed_size = 64):
    inputs = keras.Input(shape=(None,))
    embedding = keras.layers.Embedding(input_dim = vocab_size + num_oov_buckets, 
                                       output_dim = embed_size, 
                                       mask_zero = True, 
                                       input_shape = [None])(inputs)
    query = keras.layers.Conv1D(filters=100, kernel_size=4, padding="same")(embedding)
    value = keras.layers.Conv1D(filters=100, kernel_size=4, padding="same")(embedding)
    attention = keras.layers.Attention()([query, value])
    concat = keras.layers.Concatenate()([query, attention])
    lstm1 = keras.layers.LSTM(embed_size, return_sequences = True)(concat)
    bn = keras.layers.BatchNormalization()(lstm1)
    lstm2 = keras.layers.LSTM(embed_size)(bn)
    outputs = keras.layers.Dense(1, activation = "sigmoid")(lstm2)
    
    model = keras.Model(inputs, outputs)
    
    return model


def compile_model(model):
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model


def train_model(compiled_model, train_set, valid_set, model_patience, model_epochs):
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor = 'loss', patience = model_patience)
    history = compiled_model.fit(train_set, validation_data = valid_set, epochs = model_epochs, callbacks = [early_stopping_cb])

    return history