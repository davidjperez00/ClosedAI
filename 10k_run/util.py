import tensorflow_datasets as tfds
import tensorflow as tf

def preprocess(instance):
  # Potential dimmensionality reduction (consider different shapes):
  # input_image = tf.image.resize(datapoint['image'], (128, 128))

  # Normalize the input to 0-1 range
  input_image = tf.cast(instance['image'], tf.float32) / 255.0

  # Replace pixel values equal to 255 with 0, else 1
  # 255 pixel values arn't lane lines and other pixel
  # values are neglected since other subtasks of lane marking
  # are ignored.
  bitmask_label = tf.where(instance['label'] == 255, 1, 0) 

  return (input_image, bitmask_label)

# Loads a subset of the bdd100k dataset and returns it.
# The dataset contains "train", "validate", and "test" sections
def load_10k_data():
  # Retrieve custom tfds of BDD10k datatset
  dataset = tfds.load('bdd')

  train_set = dataset['train'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
  validate_set = dataset['test'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
  test_set = dataset['validate'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
  
  return train_set, validate_set, test_set
