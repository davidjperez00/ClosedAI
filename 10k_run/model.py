import tensorflow as tf
import numpy as np
# from tensorflow_examples.models.pix2pix import pix2pix

''' 
Code pulled from tensforflow_examples:
  - upsample()
  - InstanceNomalization
  
  link: https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

'''
class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def create_ResNetV2_model():
  base_model = tf.keras.applications.ResNet50V2(input_shape=[384, 640, 3], include_top=False)

  # Use the activations of these layers
  layer_names = [
    "conv1_conv",  # (192,320,64)
    "conv2_block1_3_conv", # (96, 160, out = 256)
    "conv3_block3_1_conv", # (48, 80, out=256)
    "conv4_block1_1_conv", # (24, 40 out=512)
    "conv5_block1_1_conv", # (12,20, out=1024)
  ]
  
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

  down_stack.trainable = False

  up_stack = [
    upsample(256, 3), 
    upsample(128, 3),  
    upsample(64, 3),  
    upsample(32, 3),  
  ]

  return up_stack, down_stack

def resnet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[384, 640, 3])

  up_stack, down_stack = create_ResNetV2_model()

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  ## Testing addition of new layers
  initializer = tf.random_normal_initializer(0., 0.02)

  new_one = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=initializer, activation='relu')(x)
  batch_one = tf.keras.layers.BatchNormalization()(new_one)

  new_two = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_initializer=initializer, activation='relu')(batch_one)
  batch_two = tf.keras.layers.BatchNormalization()(new_two)

  new_three = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', kernel_initializer=initializer, activation='relu')(batch_two)
  batch_three = tf.keras.layers.BatchNormalization()(new_three)

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(batch_three)

  return tf.keras.Model(inputs=inputs, outputs=x)
  