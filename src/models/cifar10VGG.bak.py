import tensorflow as tf
import numpy as np
from option.options import opt
from utils.fftConvLayer import FFTConv

if opt.restore and opt.net=='cifar10Net':
  state = np.load(opt.restoredir+"/cifar10VGG11.npz")
  print(state.keys())
  stat = state['state'].item()
  pre_params = stat['params']

def init_weight(shape, restore, *argv):
  if restore == False: 
    if 2==len(shape):
      fan_in = shape[0]
    else:
      fan_in = shape[0]*shape[1]*shape[2]
    stddev = np.sqrt(2.0/fan_in)
    w = tf.random_normal(shape = shape, mean = 0, stddev = stddev)
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

def batch_norm(x, oC, isTrain, isRes, *argv):
  batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name=None)

  ema_mean = tf.train.ExponentialMovingAverage(decay=0.9, zero_debias=True, name='mean')
  ema_var = tf.train.ExponentialMovingAverage(decay=0.9, zero_debias=True, name='var')
    
  def mean_var_with_update():
    ema_mean_apply_op = ema_mean.apply([batch_mean])
    ema_var_apply_op = ema_var.apply([batch_var])
    with tf.control_dependencies([ema_mean_apply_op, ema_var_apply_op]):
      return tf.identity(batch_mean), tf.identity(batch_var)

  if False==isRes:
    beta = tf.Variable(tf.constant(0.0, shape=[oC]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[oC]), name='gamma', trainable=True)
  else:
    beta = tf.Variable(pre_params[argv[0]+'/beta:0'], name='beta', trainable=True)
    gamma = tf.Variable(pre_params[argv[0]+'/gamma:0'], name='gamma', trainable=True)
  mean, var = tf.cond(isTrain, mean_var_with_update,lambda:(ema_mean.average(batch_mean), ema_var.average(batch_var)))
  x_normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
  return x_normed

def cifar10Net(x, isTrain):
  with tf.variable_scope("conv1_1"):
    w_conv1_1 = tf.Variable(init_weight([3,3,3,64], opt.restore, 'conv1_1/w:0'), name='w')
    b_conv1_1 = tf.Variable(init_bias([64], opt.restore, 'conv1_1/b:0'), name='b')
    conv1_1 = tf.nn.conv2d(x,w_conv1_1, strides=[1,1,1,1],padding='SAME') + b_conv1_1
    conv1_1 = batch_norm(conv1_1, 64, isTrain, opt.restore, 'conv1_1')
    conv1_1 = tf.nn.relu(conv1_1)
    print('conv1_1: ',conv1_1.shape)

  pool1 = tf.nn.max_pool(conv1_1, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  #pool1 = tf.nn.dropout(pool1, 0.4)
  print('pool1: ', pool1.shape)

  with tf.variable_scope("conv2_1"):
    w_conv2_1 = tf.Variable(init_weight([3,3,64,128], opt.restore, 'conv2_1/w:0'), name='w')
    b_conv2_1 = tf.Variable(init_bias([128], opt.restore, 'conv2_1/b:0'), name='b')
    conv2_1 = tf.nn.conv2d(pool1,w_conv2_1, strides=[1,1,1,1],padding='SAME') + b_conv2_1
    conv2_1 = batch_norm(conv2_1, 128, isTrain, opt.restore, 'conv2_1')
    conv2_1 = tf.nn.relu(conv2_1)
    print('conv2_1: ', conv2_1.shape)

  pool2 = tf.nn.max_pool(conv2_1, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  #pool2 = tf.nn.dropout(pool2, 0.4)
  print('pool2: ', pool2.shape)

  with tf.variable_scope("conv3_1"):
    w_conv3_1 = tf.Variable(init_weight([3,3,128,256], opt.restore, 'conv3_1/w:0'), name='w')
    b_conv3_1 = tf.Variable(init_bias([256], opt.restore, 'conv3_1/b:0'), name='b')
    conv3_1 = tf.nn.conv2d(pool2, w_conv3_1, strides=[1,1,1,1],padding='SAME') + b_conv3_1
    conv3_1 = batch_norm(conv3_1, 256, isTrain, opt.restore, 'conv3_1')
    conv3_1 = tf.nn.relu(conv3_1)
    print('conv3_1: ', conv3_1.shape)

  with tf.variable_scope("conv3_2"):
    w_conv3_2 = tf.Variable(init_weight([3,3,256,256], opt.restore, 'conv3_2/w:0'), name='w')
    b_conv3_2 = tf.Variable(init_bias([256], opt.restore, 'conv3_2/b:0'), name='b')
    conv3_2 = tf.nn.conv2d(conv3_1, w_conv3_2, strides=[1,1,1,1],padding='SAME') + b_conv3_2
    conv3_2 = batch_norm(conv3_2, 256, isTrain, opt.restore, 'conv3_2')
    conv3_2 = tf.nn.relu(conv3_2)
    print('conv3_2: ', conv3_2.shape)

  pool3 = tf.nn.max_pool(conv3_2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print('pool3: ', pool3.shape)

  with tf.variable_scope("conv4_1"):
    w_conv4_1 = tf.Variable(init_weight([3,3,256,512], opt.restore, 'conv4_1/w:0'), name='w')
    b_conv4_1 = tf.Variable(init_bias([512], opt.restore, 'conv4_1/b:0'), name='b')
    conv4_1 = tf.nn.conv2d(pool3, w_conv4_1, strides=[1,1,1,1],padding='SAME') + b_conv4_1
    conv4_1 = batch_norm(conv4_1, 512, isTrain, opt.restore, 'conv4_1')
    conv4_1 = tf.nn.relu(conv4_1)
    print('conv4_1: ', conv4_1.shape)

  with tf.variable_scope("conv4_2"):
    w_conv4_2 = tf.Variable(init_weight([3,3,512,512], opt.restore, 'conv4_2/w:0'), name='w')
    b_conv4_2 = tf.Variable(init_bias([512], opt.restore, 'conv4_2/b:0'), name='b')
    conv4_2 = tf.nn.conv2d(conv4_1, w_conv4_2, strides=[1,1,1,1],padding='SAME') + b_conv4_2
    conv4_2 = batch_norm(conv4_2, 512, isTrain, opt.restore, 'conv4_2')
    conv4_2 = tf.nn.relu(conv4_2)
    print('conv4_2: ', conv4_2.shape)

  pool4 = tf.nn.max_pool(conv4_2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print('pool4: ', pool4.shape)

  with tf.variable_scope("conv5_1"):
    w_conv5_1 = tf.Variable(init_weight([3,3,512,512], opt.restore, 'conv5_1/w:0'), name='w')
    b_conv5_1 = tf.Variable(init_bias([512], opt.restore, 'conv5_1/b:0'), name='b')
    conv5_1 = tf.nn.conv2d(pool4, w_conv5_1, strides=[1,1,1,1],padding='SAME') + b_conv5_1
    conv5_1 = batch_norm(conv5_1, 512, isTrain, opt.restore, 'conv5_1')
    conv5_1 = tf.nn.relu(conv5_1)
    print('conv5_1: ', conv5_1.shape)

  with tf.variable_scope("conv5_2"):
    w_conv5_2 = tf.Variable(init_weight([3,3,512,512], opt.restore, 'conv5_2/w:0'), name='w')
    b_conv5_2 = tf.Variable(init_bias([512], opt.restore, 'conv5_2/b:0'), name='b')
    conv5_2 = tf.nn.conv2d(conv5_1, w_conv5_2, strides=[1,1,1,1],padding='SAME') + b_conv5_2
    conv5_2 = batch_norm(conv5_2, 512, isTrain, opt.restore, 'conv5_2')
    conv5_2 = tf.nn.relu(conv5_2)
    print('conv5_2: ', conv5_2.shape)

  pool5 = tf.nn.max_pool(conv5_2, 
                         ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
  print('pool4: ', pool5.shape)

  conv_flatten = tf.layers.flatten(pool5)
  #conv_flatten = tf.nn.dropout(conv_flatten, 0.6)


  #with tf.name_scope("fc1"):
  #  w_fc1 = tf.Variable(init_weight([512, 512], opt.restore, 'fc1/w:0'), name='w')
  #  b_fc1 = tf.Variable(init_weight([512], opt.restore, 'fc1/b:0'), name='b')
  #  fc1 = tf.matmul(conv_flatten, w_fc1) + b_fc1
  #  fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, 0.6)

  #with tf.name_scope("fc2"):
  #  fc2 = Dense(512, 'fc2')(fc1)
    #fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, 0.5)
  
  with tf.variable_scope('fc3'):
    w_fc3 = tf.Variable(init_weight([512, 10], opt.restore, 'fc3/w:0'), name='w')
    b_fc3 = tf.Variable(init_bias([10], opt.restore, 'fc3/b:0'), name='b')
    fc3 = tf.matmul(conv_flatten, w_fc3) + b_fc3

  return fc3

def cifar10FFT(x, Budget):
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


