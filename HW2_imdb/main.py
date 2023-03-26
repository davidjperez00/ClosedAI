import tensorflow as tf
from tensorflow import keras
import numpy as np

import util
import model as model_file

MODEL_ONE_SAVE_PATH = 'GRU_model_1'
MODEL_ONE_HISTORY_SAVE_PATH = 'GRU_model_1_history.pkl'

MODEL_ONE_ALT_SAVE_PATH = 'GRU_model_1_alt'
MODEL_ONE_ALT_HISTORY_SAVE_PATH = 'GRU_model_1_alt_history.pkl'

MODEL_TWO_SAVE_PATH = 'LSTM_model_1'
MODEL_TWO_HISTORY_SAVE_PATH = 'LSTM_model_1_history.pkl'


def main():
    keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    train_first_model()
    train_first_model_alt()
    train_second_model()
    
    
def train_first_model():
    num_oov_buckets = 1000
    vocab_size = 10000
    embed_size = 128
    
    datasets = util.load_imdb_data()
    vocabulary = util.build_vocabulary(datasets)
    truncated_vocabulary = util.truncat_vocab(vocabulary, vocab_size)
    train_set, valid_set, test_set  = util.create_imdb_lookup_table(truncated_vocabulary, num_oov_buckets, datasets)
    
    gru_model_one = model_file.create_GRU_model(vocab_size, num_oov_buckets, embed_size)
    compiled_gru_model_one = model_file.compile_model(gru_model_one)
    gru_model_one_history = model_file.train_model(compiled_gru_model_one, train_set, valid_set, 5, 20)
    
    # # Save model training history to be used later for plotting training
    util.save_pkl_model_history(gru_model_one_history, MODEL_ONE_HISTORY_SAVE_PATH)

    # # Save our model for later analysis
    util.save_keras_model(compiled_gru_model_one, MODEL_ONE_SAVE_PATH)
    
def train_first_model_alt():
    num_oov_buckets = 1000
    vocab_size = 10000
    embed_size = 128
    
    datasets = util.load_imdb_data()
    vocabulary = util.build_vocabulary(datasets)
    truncated_vocabulary = util.truncat_vocab(vocabulary, vocab_size)
    train_set, valid_set, test_set  = util.create_imdb_lookup_table(truncated_vocabulary, num_oov_buckets, datasets)
    
    gru_model_one_alt = model_file.create_GRU_model_alt(vocab_size, num_oov_buckets, embed_size)
    compiled_gru_model_one_alt = model_file.compile_model(gru_model_one_alt)
    gru_model_one_alt_history = model_file.train_model(compiled_gru_model_one_alt, train_set, valid_set, 5, 20)
    
    # # Save model training history to be used later for plotting training
    util.save_pkl_model_history(gru_model_one_alt_history, MODEL_ONE_ALT_HISTORY_SAVE_PATH)

    # # Save our model for later analysis
    util.save_keras_model(compiled_gru_model_one_alt, MODEL_ONE_ALT_SAVE_PATH)
        
    
def train_second_model():
    num_oov_buckets = 1000
    vocab_size = 10000
    embed_size = 128
    
    datasets = util.load_imdb_data()
    vocabulary = util.build_vocabulary(datasets)
    truncated_vocabulary = util.truncat_vocab(vocabulary, vocab_size)
    train_set, valid_set, test_set  = util.create_imdb_lookup_table(truncated_vocabulary, num_oov_buckets, datasets)
    
    lstm_model = model_file.create_LSTM_model_1(vocab_size, num_oov_buckets, embed_size)
    lstm_model = model_file.compile_model(lstm_model)
    lstm_history = model_file.train_model(lstm_model, train_set, valid_set, 5, 20)
    
    # Save model and model history
    util.save_pkl_model_history(lstm_history, MODEL_TWO_HISTORY_SAVE_PATH)
    util.save_keras_model(lstm_model, MODEL_TWO_SAVE_PATH)
    
    

if __name__ == '__main__':
    main()
        

