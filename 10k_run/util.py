import tensorflow_datasets as tfds
import tensorflow as tf
import pickle

def preprocess(instance):
  # Resize each image from (720, 1280) -> (384, 640)
  resized = tf.image.resize(instance['image'], (384, 640))

  # Normalize the input to 0-1 range
  input_image = tf.cast(resized, tf.float32) / 255.0

  # Resize bitmask's from (720, 1280) -> (384, 640):
  resized_bitmask = tf.image.resize(instance['label'], (384, 640))

  # Replace pixel values equal to 255 with 0, else 1.
  # 255 pixel values arn't lane lines and other non 255
  # pixel's represent other lane line attributes. Our approach
  # is to simply identify a lane line or not, hence all lane 
  # lines are set to a value of 1.
  bitmask_label = tf.where(resized_bitmask == 255, 0, 1) 
  
  return (input_image, bitmask_label)

# Loads a subset of the bdd100k dataset and returns it.
# The dataset contains "train", "validate", and "test" sections
def load_10k_data():
  # Parial implementation for new model testing purposes
#   train, valid = tfds.load('bdd', split=['train[:2000]', 'test[:2000]'])
#   train_set = train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(700).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
#   validate_set = valid.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(700).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

  # Full implementation:
  dataset = tfds.load('bdd')
  train_set = dataset['train'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
  # This should have called validation portion of dataset but is left now
  # Since our model is trainined with this implemenation
  validate_set = dataset['test'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

  # Unused for training and help off for testing
  # test_set = dataset['validate'].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(800).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
  
  return train_set, validate_set

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