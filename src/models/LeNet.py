import tensorflow as tf
import numpy as np
from option.options import opt
from utils.fftConvLayer import FFTConv

if opt.restore:
  pre_params = np.load(opt.modelfile)

def init_weight(shape, restore, *argv):
  if restore == False: 
    w = tf.random_normal(shape = shape, mean = 0, stddev = 0.1)
  else:
    w = pre_params[argv[0]]
  return w

def init_bias(shape, restore, *argv):
  if restore == False:
    b = tf.zeros(shape)
  else:
    b = pre_params[argv[0]]
  return b

class Dense(tf.keras.layers.Layer):
  def __init__(self, shape_output, scope):
    super(Dense, self).__init__()
    self.shape_output = shape_output
    self.scope = scope

  def build(self, shape_input):
    self.kernel = tf.Variable(init_weight([shape_input[-1].value, self.shape_output], opt.restore, self.scope + "_w"), name='w')
    self.bias = tf.Variable(init_bias([self.shape_output], opt.restore, self.scope+"_b"), name='b')

  def call(self, input):
    return tf.matmul(input, self.kernel)+self.bias

def LeNet_init(self):
  self.budget = [64, 64]
  self.imagsize = 28
  self.imagpad = 2
  self.insize = 32
  self.inchnl = 1
  self.tf_x = tf.placeholder(tf.float32, (None, 32, 32, 1))
  self.tf_y = tf.placeholder(tf.int32, (None))
  self.one_hot_y = tf.one_hot(self.tf_y, 10)
  self.dataset = tf.keras.datasets.mnist

def LeNet(self, x):
  with tf.variable_scope("conv1"):
    w_conv1 = tf.Variable(init_weight([5,5,1,6], opt.restore, 'conv1_w'), name='w')
    b_conv1 = tf.Variable(init_bias([6], opt.restore, 'conv1_b'), name='b')
    conv1 = tf.nn.conv2d(x,w_conv1, strides=[1,1,1,1],padding='VALID') + b_conv1
    conv1 = tf.nn.relu(conv1)
    print(conv1.shape)

  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print(pool1.shape)
  with tf.variable_scope("conv2"):
    w_conv2 = tf.Variable(init_weight([5,5,6,16], opt.restore, 'conv2_w'), name='w')
    b_conv2 = tf.Variable(init_bias([16], opt.restore, 'conv2_b'), name='b')
    conv2 = tf.nn.conv2d(pool1, w_conv2, strides=[1,1,1,1],padding='VALID') + b_conv2
    conv2 = tf.nn.relu(conv2)
    print(conv2.shape)

  pool2 = tf.nn.max_pool(conv2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print(pool2.shape)
  conv_flatten = tf.layers.flatten(pool2)

  with tf.name_scope("fc1"):
    fc1 = Dense(120, 'fc1')(conv_flatten)
    fc1 = tf.nn.relu(fc1)
  
  with tf.variable_scope('fc2'):
    fc2 = Dense(84, 'fc2')(fc1)
    fc2 = tf.nn.relu(fc2)

  with tf.variable_scope('fc3'):
    fc3 = Dense(10, 'fc3')(fc2)

  return fc3, 'fc1'

def LeNetFFT(x, Budget):
  wconv1 = init_weight([5,5,1,6], opt.restore, 'conv1_w')
  bconv1 = init_bias([6], opt.restore, 'conv1_b')
  conv1 = FFTConv(opt.fftsize, 12, wconv1, bconv1, Budget[0])(x)
  conv1 = tf.nn.relu(conv1)

  pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

  wconv2 = init_weight([5,5,6,16], opt.restore, 'conv2_w')
  bconv2 = init_bias([16], opt.restore, 'conv2_b')
  conv2 = FFTConv(opt.fftsize, 7, wconv2, bconv2, Budget[1])(pool1)
  conv2 = tf.nn.relu(conv2)

  pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

  conv_flatten = tf.layers.flatten(pool2)

  fc1 = Dense(120,'fc1')(conv_flatten)
  fc1 = tf.nn.relu(fc1)

  fc2 = Dense(84, 'fc2')(fc1)
  fc2 = tf.nn.relu(fc2)

  fc3 = Dense(10, 'fc3')(fc2)

  return fc3


