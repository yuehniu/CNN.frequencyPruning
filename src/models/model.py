import tensorflow as tf
import numpy as np
from hyperparam import BATCHSIZE

def init_weight(shape):
  w = tf.random_normal(shape = shape, mean = 0, stddev = 0.1)
  return tf.Variable(w)

def init_bias(shape):
  b = tf.zeros(shape)
  return tf.Variable(b)

class Dense(tf.keras.layers.Layer):
  def __init__(self, shape_output):
    super(Dense, self).__init__()
    self.shape_output = shape_output

  def build(self, shape_input):
    self.kernel = init_weight([shape_input[-1].value, self.shape_output]) 
    self.bias = init_bias([self.shape_output])

  def call(self, input):
    print(input.shape)
    print(self.bias.shape)
    return tf.matmul(input, self.kernel)+self.bias

class FFTConv(tf.keras.layers.Layer):
  def __init__(self, channel_output):
    super(FFTConv, self).__init__()
    self.channel_output = channel_output

  def build(self, shape_input):
    # self.shape_kern = shape_input[1].value
    self.kernel = tf.random_normal(shape = \
                                          [self.channel_output, 
                                           shape_input[1].value, 
                                           shape_input[2].value,
                                           shape_input[2].value])
    self.kernel_fft = tf.spectral.fft2d(tf.cast(self.kernel, tf.complex64))
    self.kernel_real = tf.Variable(tf.real(self.kernel_fft))
    self.kernel_imag = tf.Variable(tf.imag(self.kernel_fft))
    self.bias = init_bias([self.channel_output])
    self.kernel_freq = tf.complex(self.kernel_real, self.kernel_imag)
  
  def call(self, input):
    x_fft = tf.spectral.fft2d(tf.cast(input, tf.complex64))
    #y_accum = tf.placeholder(tf.float32, 
    #  (input.shape[0].value,
    #   self.channel_output, 
    #   input.shape[-1].value, 
    #   input.shape[-1].value))
    y_fft = tf.multiply(self.kernel_freq, x_fft[0,:])
    y_fft_accum = tf.reduce_sum(y_fft, 1)
    yb = tf.spectral.ifft2d(y_fft_accum)
    bias_expand = tf.expand_dims(tf.expand_dims(self.bias, 1),1)
    y_accum = tf.expand_dims(tf.real(yb) + bias_expand, 0)
    for b in range(1,BATCHSIZE):
      y_fft = tf.multiply(self.kernel_freq, x_fft[b,:])
      y_fft_accum = tf.reduce_sum(y_fft, 1)
      yb = tf.spectral.ifft2d(y_fft_accum)
      bias_expand = tf.expand_dims(tf.expand_dims(self.bias, 1),1)
      y_accum = tf.concat([y_accum, tf.expand_dims(tf.real(yb) + bias_expand, 0)],0)
    return y_accum

def LeNet(x):
  conv1 = FFTConv(6)(x)
  conv1 = tf.nn.relu(conv1)
  print(conv1.shape)
  conv1 = tf.transpose(conv1, perm=[0,2,3,1])
  print(conv1.shape)
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print(pool1.shape)
  pool1 = tf.transpose(pool1, perm=[0,3,1,2])
  print(pool1.shape)
  conv2 = FFTConv(16)(pool1)
  conv2 = tf.nn.relu(conv2)
  print(conv2.shape)
  conv2 = tf.transpose(conv2, perm=[0,2,3,1])
  print(conv2.shape)
  pool2 = tf.nn.max_pool(conv2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print(pool2.shape)
  conv_flatten = tf.layers.flatten(pool2)

  fc1 = Dense(120)(conv_flatten)
  fc1 = tf.nn.relu(fc1)
  
  fc2 = Dense(84)(fc1)
  fc2 = tf.nn.relu(fc2)

  fc3 = Dense(10)(fc2)

  return fc3
