import tensorflow as tf
import numpy as np
from option.options import opt
from utils.fftConvLayer import FFTConv

if opt.restore and opt.net=='cifar10Net':
  pre_params = np.load(opt.restoredir+"/cifar10NetPretrained.npz")

def init_weight(shape, restore, *argv):
  if restore == False: 
    w = tf.random_normal(shape = shape, mean = 0, stddev = 0.1)
    #w = tf.random_uniform(shape = shape)
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

def batch_norm(x, oC, isTrain, isRes=False):
  if False==isRes:
    beta = tf.Variable(tf.constant(0.0, shape=[oC]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[oC]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(isTrain, mean_var_with_update,lambda:(ema.average(batch_mean), ema.average(batch_var)))
    x_normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return x_normed

def cifar10Net(x, isTrain):
  with tf.variable_scope("conv1_1"):
    w_conv1_1 = tf.Variable(init_weight([3,3,3,64], opt.restore, 'conv1_1_w'), name='w')
    b_conv1_1 = tf.Variable(init_bias([64], opt.restore, 'conv1_1_b'), name='b')
    conv1_1 = tf.nn.conv2d(x,w_conv1_1, strides=[1,1,1,1],padding='SAME') + b_conv1_1
    conv1_1 = batch_norm(conv1_1, 64, isTrain)
    conv1_1 = tf.nn.relu(conv1_1)
    print('conv1_1: ',conv1_1.shape)

  with tf.variable_scope("conv1_2"):
    w_conv1_2 = tf.Variable(init_weight([3,3,64,64], opt.restore, 'conv1_2_w'), name='w')
    b_conv1_2 = tf.Variable(init_bias([64], opt.restore, 'conv1_2_b'), name='b')
    conv1_2 = tf.nn.conv2d(conv1_1,w_conv1_2, strides=[1,1,1,1],padding='SAME') + b_conv1_2
    conv1_2 = batch_norm(conv1_2, 64, isTrain)
    conv1_2 = tf.nn.relu(conv1_2)
    print('conv1_2: ',conv1_2.shape)

  pool1 = tf.nn.max_pool(conv1_2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  #pool1 = tf.nn.dropout(pool1, 0.4)
  print('pool1: ', pool1.shape)

  with tf.variable_scope("conv2_1"):
    w_conv2_1 = tf.Variable(init_weight([3,3,64,128], opt.restore, 'conv2_1_w'), name='w')
    b_conv2_1 = tf.Variable(init_bias([128], opt.restore, 'conv2_1_b'), name='b')
    conv2_1 = tf.nn.conv2d(pool1,w_conv2_1, strides=[1,1,1,1],padding='SAME') + b_conv2_1
    conv2_1 = batch_norm(conv2_1, 128, isTrain)
    conv2_1 = tf.nn.relu(conv2_1)
    print('conv2_1: ', conv2_1.shape)

  with tf.variable_scope("conv2_2"):
    w_conv2_2 = tf.Variable(init_weight([3,3,128,128], opt.restore, 'conv2_2_w'), name='w')
    b_conv2_2 = tf.Variable(init_bias([128], opt.restore, 'conv2_2_b'), name='b')
    conv2_2 = tf.nn.conv2d(conv2_1,w_conv2_2, strides=[1,1,1,1],padding='SAME') + b_conv2_2
    conv2_2 = batch_norm(conv2_2, 128, isTrain)
    conv2_2 = tf.nn.relu(conv2_2)
    print('conv2_2: ', conv2_2.shape)

  pool2 = tf.nn.max_pool(conv2_2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  #pool2 = tf.nn.dropout(pool2, 0.4)
  print('pool2: ', pool2.shape)

  with tf.variable_scope("conv3_1"):
    w_conv3_1 = tf.Variable(init_weight([3,3,128,256], opt.restore, 'conv3_1_w'), name='w')
    b_conv3_1 = tf.Variable(init_bias([256], opt.restore, 'conv3_1_b'), name='b')
    conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides=[1,1,1,1],padding='SAME') + b_conv3_1
    conv3_1 = batch_norm(conv3_1, 256, isTrain)
    conv3_1 = tf.nn.relu(conv3_1)
    print('conv3_1: ', conv3_1.shape)

  with tf.variable_scope("conv3_2"):
    w_conv3_2 = tf.Variable(init_weight([3,3,256,256], opt.restore, 'conv3_2_w'), name='w')
    b_conv3_2 = tf.Variable(init_bias([256], opt.restore, 'conv3_2_b'), name='b')
    conv3_2 = tf.nn.conv2d(conv3_1, w_conv3_2, strides=[1,1,1,1],padding='SAME') + b_conv3_2
    conv3_2 = batch_norm(conv3_2, 256, isTrain)
    conv3_2 = tf.nn.relu(conv3_2)
    print('conv3_2: ', conv3_2.shape)

  with tf.variable_scope("conv3_3"):
    w_conv3_3 = tf.Variable(init_weight([3,3,256,256], opt.restore, 'conv3_3_w'), name='w')
    b_conv3_3 = tf.Variable(init_bias([256], opt.restore, 'conv3_3_b'), name='b')
    conv3_3 = tf.nn.conv2d(conv3_2, w_conv3_3, strides=[1,1,1,1],padding='SAME') + b_conv3_3
    conv3_4 = batch_norm(conv3_3, 256, isTrain)
    conv3_3 = tf.nn.relu(conv3_3)
    print('conv3_3: ', conv3_3.shape)

  with tf.variable_scope("conv3_4"):
    w_conv3_4 = tf.Variable(init_weight([3,3,256,256], opt.restore, 'conv3_4_w'), name='w')
    b_conv3_4 = tf.Variable(init_bias([256], opt.restore, 'conv3_4_b'), name='b')
    conv3_4 = tf.nn.conv2d(conv3_3, w_conv3_4, strides=[1,1,1,1],padding='SAME') + b_conv3_4
    conv3_4 = batch_norm(conv3_4, 256, isTrain)
    conv3_4 = tf.nn.relu(conv3_4)
    print('conv3_4: ', conv3_4.shape)

  pool3 = tf.nn.max_pool(conv3_4, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print('pool5: ', pool3.shape)
  conv_flatten = tf.layers.flatten(pool3)
  #conv_flatten = tf.nn.dropout(conv_flatten, 0.6)


  with tf.name_scope("fc1"):
    fc1 = Dense(512, 'fc1')(conv_flatten)
    fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, 0.6)

  #with tf.name_scope("fc2"):
  #  fc2 = Dense(512, 'fc2')(fc1)
    #fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, 0.5)
  
  with tf.variable_scope('fc3'):
    fc3 = Dense(10, 'fc3')(fc1)

  return fc3

