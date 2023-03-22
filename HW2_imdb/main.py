
import tensorflow as tf
from tensorflow import keras
import numpy as np

import util
import model as model_file

MODEL_ONE_SAVE_PATH = 'GRU_model_1'
MODEL_ONE_HISTORY_SAVE_PATH = 'GRU_model_1_history.pkl"'
MODEL_TWO_SAVE_PATH = 'cifar100_model_2'
MODEL_TWO_HISTORY_SAVE_PATH = 'cifar100_model_1_history.pkl"'

def main():
  keras.backend.clear_session()
  tf.random.set_seed(42)
  np.random.seed(42)

  num_oov_buckets = 1000
  vocab_size = 10000
  embed_size = 128
  
  datasets = util.load_imdb_data()
  
  vocabulary = util.build_vocabulary(datasets)

  truncated_vocabulary = util.truncat_vocab(vocabulary, vocab_size)

  train_set, test_set = util.create_imdb_lookup_table(truncated_vocabulary, num_oov_buckets, datasets)
  
  gru_model_one = model_file.create_GRU_model(vocab_size, num_oov_buckets, embed_size)

  compiled_gru_model_one = model_file.compile_GRU_model(gru_model_one)

  gru_model_one_history = model_file.train_GRU_model(compiled_gru_model_one, train_set)

  # # Save model training history to be used later for plotting training
  util.save_pkl_model_history(gru_model_one_history, MODEL_ONE_HISTORY_SAVE_PATH)

  # # Save our model for later analysis
  util.save_keras_model(compiled_gru_model_one, MODEL_ONE_SAVE_PATH)

if __name__ == '__main__':
  main()
    
