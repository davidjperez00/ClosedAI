import tensorflow_datasets as tfds
from collections import Counter
import tensorflow as tf
import pickle


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

    # A clumbsy workaround to allow the use of the map function 
    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    # Creating encodings for training set
    train_set = datasets["train"].batch(32).map(preprocess)
    train_set = train_set.map(encode_words)

    test_set = datasets["test"]
    valid_set = test_set.take(10000)
    test_set = test_set.skip(10000)

    # Creating encodings for validation set
    valid_set = valid_set.batch(32).map(preprocess)
    valid_set = valid_set.map(encode_words)

    # Creating encodings for test set
    test_set = test_set.batch(32).map(preprocess)
    test_set = test_set.map(encode_words)

    return train_set, valid_set, test_set 

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