import tensorflow as tf
from tensorflow import keras
import numpy as np

import util
import model as model_file

def main():
  keras.backend.clear_session()
  tf.random.set_seed(42)
  np.random.seed(42)

  ''' TRANING FIRST MODEL '''
  train_set, validate_set = util.load_10k_data()

  model = model_file.resnet_model(output_channels=2)

  ad_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Default is 0.001
  model.compile(optimizer=ad_optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
  model_history = model.fit(train_set, epochs=45,
                          validation_data=validate_set,
                          callbacks=[early_stopping_cb])

  util.save_pkl_model_history(model_history, 'model_d11_3new_full_45_10pat_history.pkl')
                          
  util.save_keras_model(model, 'model_d11_3new_full_45e_10pat')


if __name__ == '__main__':
  main()
