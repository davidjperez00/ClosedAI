# import tensorflow_datasets as tfds
# from collections import Counter
# import tensorflow as tf
from tensorflow import keras

def create_GRU_model(vocab_size, num_oov_buckets, embed_size = 128):
    # embed_size = 128
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, mask_zero = True, input_shape = [None]),
        keras.layers.GRU(128, return_sequences = True),
        keras.layers.BatchNormalization(),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation = "sigmoid")
    ])

    return model


def create_LSTM_model_1(vocab_size, num_oov_buckets, embed_size = 64):
    model = keras.models.Sequential([
        keras.layers.Embedding(input_dim = vocab_size + num_oov_buckets, output_dim = embed_size, mask_zero = True, input_shape = [None]),
        keras.layers.LSTM(embed_size, return_sequences = True),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(embed_size),
        keras.layers.Dense(1, activation = "sigmoid")
    ])
    
    return model


def compile_model(model):
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model


def train_model(compiled_model, train_set, valid_set, model_patience, model_epochs):
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor = 'loss', patience = model_patience)
    history = compiled_model.fit(train_set, validation_data = valid_set, epochs = model_epochs, callbacks = [early_stopping_cb])

    return history