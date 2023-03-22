import tensorflow_datasets as tfds
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from itertools import repeat

# To allow for for map function in 'create_preprocessed_train_set()'
# the lookup table is made global to still allow for the effeciency
# increase from the .prefetch() method.
GRU_LOOKUP_TABLE = None

def load_imdb_data():
  datasets = tfds.load("imdb_reviews", as_supervised=True)

  return datasets

# @Brief: Preprocessing for unwanted/undesirable review data.
#
# @Note: Truncates each dataset instance to a max of 300 words.abs
#        - Regular expression is used to replace '<br />' tags with spaces,
#          and replaces any characters other than letters and quotes with spaces.
#        - Finally the reviews are split by the spaces, which returns a ragged
#          tensor and converts it to a dense tensor, padding all reviews with padding
#          token '<pad>' so that they all have the same length.
def preprocess(X_batch, y_batch):
  X_batch = tf.strings.substr(X_batch, 0, 300)
  X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
  X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
  X_batch = tf.strings.split(X_batch)
  
  return X_batch.to_tensor(default_value=b"<pad>"), y_batch

# @brief: Creating a vocabulary of all the words contained in dataset
def build_vocabulary(datasets):
  vocabulary = Counter()

  for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
      vocabulary.update(list(review.numpy()))

  for X_batch, y_batch in datasets["test"].batch(32).map(preprocess):
    for review in X_batch:
      vocabulary.update(list(review.numpy()))
  
  return vocabulary

# @Brief: Truncating the our vocabulary to only the 10,000 most common words.
#
# @Note: that there were 53,893 but many of those are rare occurances and 
#     don't have much effect on the models performance (and 
#     increases training time).
def truncat_vocab(vocabulary, vocab_size=10000):
  truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

  return truncated_vocabulary

# @Brief: Replacing each word with its ID (i.e., its index in the vocabulary)
def create_imdb_lookup_table(truncated_vocabulary, num_oov_buckets, datasets):
  words = tf.constant(truncated_vocabulary) # all words from dataset
  word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64) # creating id's for our words
  vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
  # oov buckets are used to store extra words that don't exist within out dataset
  table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

  # A clumbsy workaround to allow the use of the map function with prefetch
  def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

  # Creating encodings for training set
  train_set = datasets["train"].batch(32).map(preprocess)
  train_set = train_set.map(encode_words).prefetch(1)

  # Creating encodings for test set
  test_set = datasets["test"].batch(32).map(preprocess)
  test_set = test_set.map(encode_words).prefetch(1)

  return train_set, test_set


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