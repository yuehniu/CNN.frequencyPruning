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
  def __init__(self, channel_output, kernel_size, tile_num):
    super(FFTConv, self).__init__()
    self.channel_output = channel_output
    self.kernel_size = kernel_size
    self.tile_num = tile_num

  def build(self, shape_input):
    """Setup fft weights"""
    # shape_input[0]: batch size
    # shape_input[1],[2]: img_size
    # shape_input[3]: input channel
    self.img_size = shape_input[2].value
    self.tile_size = self.img_size / self.tile_num
    self.fft_size = self.kernel_size + self.tile_size - 1
    self.kernel_init = tf.random_normal(shape = \
                         [self.channel_output, 
                          shape_input[3].value, 
                          self.fft_size,
                          self.fft_size])
    self.kernel_fft = tf.spectral.fft2d(tf.cast(self.kernel_init, tf.complex64))
    self.kernel_real = tf.Variable(tf.real(self.kernel_fft))
    self.kernel_imag = tf.Variable(tf.imag(self.kernel_fft))
    self.bias = init_bias([self.channel_output])
    self.kernel_freq = tf.complex(self.kernel_real, self.kernel_imag)
  
  def call(self, input):
    """Do frequency-domain conv on each tiles, then concat then to get original size"""
    for r in range(self.tile_num):
      for c in range(self.tile_num):
        # do frequency conv on each tile
        offset = [[r*self.tile_size+self.tile_size/2, c*self.tile_size+self.tile_size/2] for i in range(BATCHSIZE)]
        input_tile = tf.image.extract_glimpse(input, 
          [self.tile_size, self.tile_size],
          offset, centered=False, normalized=False)   
        pad_pixels = (self.fft_size - self.tile_size) / 2
        input_tile = tf.image.pad_to_bounding_box(
          input_tile, pad_pixels, pad_pixels, self.fft_size, self.fft_size)

        input_tile = tf.transpose(input_tile, perm=[0,3,1,2])
        input_fft = tf.spectral.fft2d(tf.cast(input_tile, tf.complex64))
        output_fft = tf.multiply(self.kernel_freq, input_fft[0,:])
        output_fft_accum = tf.reduce_sum(output_fft, 1)
        output_batch_i = tf.spectral.ifft2d(output_fft_accum)
        bias_expand = tf.expand_dims(tf.expand_dims(self.bias, 1),1)
        output_tile_accum = tf.expand_dims(tf.real(output_batch_i) + bias_expand, 0)
        for b in range(1,BATCHSIZE):
          output_fft = tf.multiply(self.kernel_freq, input_fft[b,:])
          output_fft_accum = tf.reduce_sum(output_fft, 1)
          output_fft_batch_i = tf.spectral.ifft2d(output_fft_accum)
          bias_expand = tf.expand_dims(tf.expand_dims(self.bias, 1),1)
          output_tile_accum = tf.concat([output_tile_accum, 
            tf.expand_dims(tf.real(output_fft_batch_i) + bias_expand, 0)],0)

        # Concat col tiles
        output_accum_col = output_tile_accum
        if c != 0:
          overlap = output_accum_col[:,:,:,-pad_pixels:] + output_tile_accum[:,:,:,0:pad_pixels]
          output_accum_col = tf.concat([output_accum_col[:,:,:,0:-pad_pixels], 
            overlap, 
            output_tile_accum[:,:,:,pad_pixels:]], 
            3)
    # Concat tow output tiles
    output_accum = output_accum_col
    if r != 0:
      overlap = output_accum[:,:,-pad_pixels:,:] + output_accum_col[:,:,0:pad_pixels,:]
      output_accum = tf.concat([output_accum[:,:,0:-pad_pixels,:], 
        overlap, 
        output_accum_col[:,:,pad_pixels:,:]], 
        2)

    output_accum = tf.transpose(output_accum, perm=[0,2,3,1])
    return tf.image.crop_to_bounding_box(output_accum, 0, 0, self.img_size, self.img_size)

def LeNet(x):
  conv1 = FFTConv(6, 5, 2)(x)
  conv1 = tf.nn.relu(conv1)
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  conv2 = FFTConv(16, 5, 1)(pool1)
  conv2 = tf.nn.relu(conv2)
  pool2 = tf.nn.max_pool(conv2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  conv_flatten = tf.layers.flatten(pool2)

  fc1 = Dense(120)(conv_flatten)
  fc1 = tf.nn.relu(fc1)
  
  fc2 = Dense(84)(fc1)
  fc2 = tf.nn.relu(fc2)

  fc3 = Dense(10)(fc2)

  return fc3
