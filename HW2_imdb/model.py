import tensorflow_datasets as tfds
from collections import Counter
import tensorflow as tf
from tensorflow import keras

def create_GRU_model(vocab_size, num_oov_buckets, embed_size=128):
  # embed_size = 128
  model = keras.models.Sequential([
      keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                            mask_zero=True, # not shown in the book
                            input_shape=[None]),
      keras.layers.GRU(128, return_sequences=True),
      keras.layers.GRU(128),
      keras.layers.Dense(1, activation="sigmoid")
  ])

  return model

def compile_GRU_model(model):
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

  return model

def train_GRU_model(compiled_model, train_set):
  history = compiled_model.fit(train_set, epochs=5)

  return history 