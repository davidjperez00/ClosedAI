import tensorflow_datasets as tfds
import tensorflow as tf


# Loads a subset of the bdd100k dataset and returns it.
# The dataset contains "train", "validate", and "test" sections
def load_10k_data():
    dataset = tfds.load('bdd')
    
    return dataset

# train_set = dataset["train"]
# validate_set = dataset["validate"]
# test_set = dataset["test"]