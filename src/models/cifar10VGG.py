import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() #add compatibility for tensorflow 1.*

import numpy as np
from option.options import opt
from utils.fftConvLayer import FFTConv

"""
  Note:
  This .py code is going to be used in:
  * Original net training
  * ADMM training
  Hence, codes must combine these two scenarios. 
"""

if opt.restore and opt.net=='cifar10Net':
  state = np.load(opt.modelfile, allow_pickle=True)
  print(state.keys())
  stat = state['state'].item()
  pre_params = stat['params']

def __init_weight(shape, restore, *argv):
  """ Weight initialization (Kaiming's method)
    shape: weight dimension
    restore: whether restore pre-trained weights
    argv: variable name, etc.
  """
  if restore == False: 
    if 2==len(shape):
      fan_in = shape[0]
    else:
      fan_in = shape[0]*shape[1]*shape[2]
    stddev = np.sqrt(2.0/fan_in)
    w = np.random.normal(0, stddev, shape).astype(np.float32)
    #w = tf.random_uniform(shape = shape)
  else:
    w = pre_params[argv[0]]
  return w
def __init_fweight(shape, restore, fft, *argv):
  """ Frequency-domain weight initialization
    shape: spatial weight dimension
    restore: whether load pre-trained weights
    argv: variable name, etc.
  """
  if fft:
    wreal = pre_params[argv[0]+'/kernel_real:0'] 
    wimag = pre_params[argv[0]+'/kernel_imag:0']
  elif restore:
    w_space = pre_params[argv[0]]
    # remember to flip spacial kernel when using it to initilize spectral kernels
    w_space = w_space[::-1,::-1,:,:]
    padsize = opt.fftsize - shape[0]
    pads = ((0,0),(0,0),(0,padsize),(0,padsize))
    wpad = np.pad(np.transpose(w_space, (3,2,0,1)), pads, "constant")
    w_freq = np.fft.fft2(wpad)
    wreal = np.real(w_freq)
    wimag = np.imag(w_freq)
  else:
    fan_in = opt.fftsize * shape[1] * shape[2]
    stddev = np.sqrt(2.0 / fan_in)
    fshape = [shape[2], shape[3], opt.fftsize, opt.fftsize]
    wimag = np.random.normal(0, stddev, fshape).astype(np.float32)
    wreal = np.random.normal(0, stddev, fshape).astype(np.float32)
  return wreal, wimag 
    
def __init_bias(shape, restore, *argv):
  """ Bias initialization
    shape: bias size
    restore: whether restore pre-trained bias
    argv: variable name, etc.
  """
  if restore == False:
    b = np.zeros(shape, dtype=np.float32)
  else:
    b = pre_params[argv[0]]
  return b

def __init_borm(shape, restore, layer):
  """ Initialize batch_norm layer
    shape: parameter size
    restore: whether restore pre-trained parameters
    layer: layer name
  """
  if restore:
    beta = pre_params[layer+'/batch_normalization/beta:0']
    gamma = pre_params[layer+'/batch_normalization/gamma:0']
    mean = pre_params[layer+'/batch_normalization/moving_mean:0']
    var = pre_params[layer+'/batch_normalization/moving_variance:0']
  else:
      beta = np.zeros(shape, dtype=np.float32)
      gamma = np.ones(shape, dtype=np.float32)
      mean = np.zeros(shape, dtype=np.float32)
      var = np.ones(shape, dtype=np.float32)
  return beta, gamma, mean, var

def __create_conv_layer(In, iChnl, oChnl, kDim, lyrName, Stride=1, bNorm=True, isTrain=True, isADMM=False, isFFT=False, Budget=None):
  """Conv layer creator
    In: layer input
    iChnl: input channel number
    oChnl: output channel number
    kDim: kernel size
    lyrName: conv layer name
  """
  with tf.variable_scope(lyrName[0:7]):
    if isADMM:
      bias = __init_bias([oChnl], opt.restore, lyrName[0:7]+'/bias:0')
      wreal, wimag = __init_fweight([kDim,kDim,iChnl,oChnl], opt.restore, opt.ffttrain, lyrName[0:7]+'/kernel:0') 
      f_conv = FFTConv(opt.fftsize, opt.fftsize-kDim+1, wreal, wimag, bias, Budget, Mode='same')(In)
      # wreal, wimag = __init_fweight([kDim,kDim,iChnl,oChnl], opt.restore, opt.ffttrain, lyrName+'/kernel:0')
      # w_real = tf.Variable(wreal, dtype = tf.float32, name = 'kernel_real')
      # w_imag = tf.Variable(wimag, dtype = tf.float32, name = 'kernel_imag')
      # w_freq = tf.complex(w_real, w_imag)
      # w = tf.real(tf.spectral.ifft2d(w_freq))
      # w = tf.transpose(w, perm = [0,2,3,1])
      # w = tf.image.crop_to_bounding_box(w, 0, 0, 3, 3)
      # w = tf.transpose(w, perm = [1,2,0,3])
      # f_conv = tf.nn.conv2d(In, w, strides=[1,Stride,Stride,1], padding='SAME') + bias
    elif isFFT:
      bias = __init_bias([oChnl], opt.restore, lyrName+'/bias:0')
      wreal, wimag = __init_fweight([kDim,kDim,iChnl,oChnl], opt.restore, opt.ffttrain, lyrName) 
      f_conv = FFTConv(opt.fftsize, opt.fftsize-kDim+1, wreal, wimag, bias, Budget, Mode='same')(In)
    else:
      bias = tf.Variable(__init_bias([oChnl], opt.restore, lyrName+'/bias:0'), name='bias')
      w = tf.Variable(__init_weight([kDim,kDim,iChnl,oChnl], opt.restore, lyrName+'/kernel:0'), name='kernel')
      f_conv = tf.nn.conv2d(In, w, strides=[1,Stride,Stride,1], padding='SAME') + bias
    if bNorm:
      beta,gamma,mean,var = __init_borm([oChnl], opt.restore, lyrName[0:7])
      f_conv = tf.layers.batch_normalization(f_conv, momentum=0.9, training=isTrain,
              beta_initializer=tf.constant_initializer(beta),
              gamma_initializer=tf.constant_initializer(gamma),
              moving_mean_initializer=tf.constant_initializer(mean),
              moving_variance_initializer=tf.constant_initializer(var))
    f_conv = tf.nn.relu(f_conv)
    return f_conv

def cifar10Net_init(self):
  self.budget = [int(opt.fftsize * opt.fftsize / opt.comratio)]
  self.imagsize = 32
  self.imagpad = 0
  self.insize = 32
  self.inchnl = 3
  self.tf_x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  self.tf_y = tf.placeholder(tf.int32, (None))
  self.phase = tf.placeholder(tf.bool)
  self.one_hot_y = tf.one_hot(self.tf_y, 10)
  self.dataset = tf.keras.datasets.cifar10

def cifar10Net(self, x, isTrain):
  """ Model definition
  Use higher level tensorflow API, like tf.layers.**
    x: network input
    isTrain: flag to indicate if network is in train or inference state
  """
  conv1_1 = __create_conv_layer(x, 3, 64, 3, 'conv1_1', isTrain=isTrain, isADMM=opt.admm)
  conv1_2 = __create_conv_layer(conv1_1, 64, 64, 3, 'conv1_2', isTrain=isTrain, isADMM=opt.admm)
  pool1 = tf.layers.max_pooling2d(conv1_2, [2,2], [2,2], padding='valid')
  print('pool1: ', pool1.shape)
  
  conv2_1 = __create_conv_layer(pool1, 64, 128, 3, 'conv2_1', isTrain=isTrain)
  conv2_2 = __create_conv_layer(conv2_1, 128, 128, 3, 'conv2_2', isTrain=isTrain)
  pool2 = tf.layers.max_pooling2d(conv2_2, [2,2], [2,2], padding='valid')
  print('pool2: ', pool2.shape)

  conv3_1 = __create_conv_layer(pool2, 128, 256, 3, 'conv3_1', isTrain=isTrain)
  conv3_2 = __create_conv_layer(conv3_1, 256, 256, 3, 'conv3_2', isTrain=isTrain)
  conv3_3 = __create_conv_layer(conv3_2, 256, 256, 3, 'conv3_3', isTrain=isTrain)
  pool3 = tf.layers.max_pooling2d(conv3_3, [2,2], [2,2], padding='valid')
  print('pool3: ', pool3.shape)

  conv4_1 = __create_conv_layer(pool3, 256, 512, 3, 'conv4_1', isTrain=isTrain)
  conv4_2 = __create_conv_layer(conv4_1, 512, 512, 3, 'conv4_2', isTrain=isTrain)
  conv4_3 = __create_conv_layer(conv4_2, 512, 512, 3, 'conv4_3', isTrain=isTrain)
  pool4 = tf.layers.max_pooling2d(conv4_3, [2,2], [2,2], padding='valid')
  print('pool4: ', pool4.shape)

  conv5_1 = __create_conv_layer(pool4, 512, 512, 3, 'conv5_1', isTrain=isTrain)
  conv5_2 = __create_conv_layer(conv5_1, 512, 512, 3, 'conv5_2', isTrain=isTrain)
  conv5_3 = __create_conv_layer(conv5_2, 512, 512, 3, 'conv5_3', isTrain=isTrain)
  pool5 = tf.layers.max_pooling2d(conv5_3, [2,2], [2,2], padding='valid')
  print('pool5: ', pool5.shape)

  conv_flatten = tf.layers.flatten(pool5)
  #conv_flatten = tf.nn.dropout(conv_flatten, 0.6)


  #with tf.name_scope("fc1"):
  #  w_fc1 = tf.Variable(__init_weight([512, 512], opt.restore, 'fc1/w:0'), name='w')
  #  b_fc1 = tf.Variable(__init_weight([512], opt.restore, 'fc1/b:0'), name='b')
  #  fc1 = tf.matmul(conv_flatten, w_fc1) + b_fc1
  #  fc1 = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, 0.6)

  #with tf.name_scope("fc2"):
  #  fc2 = Dense(512, 'fc2')(fc1)
    #fc2 = tf.nn.relu(fc2)
    #fc2 = tf.nn.dropout(fc2, 0.5)
  
  with tf.variable_scope('fc3'):
    ic = 512
    oc = 10
    w_fc3 = __init_weight([ic, oc], opt.restore, 'fc3/dense/kernel:0')
    b_fc3 = __init_bias([oc], opt.restore, 'fc3/dense/bias:0')
    fc3 = tf.layers.dense(conv_flatten, oc, 
          kernel_initializer=tf.constant_initializer(w_fc3),
          bias_initializer=tf.constant_initializer(b_fc3))

  return fc3, 'conv1_'

def cifar10FFT(self, x, isTrain):
  """ Post-finetune in frequency domain
    x: network input
    Budget: compression budget in each layer
  """
  conv1_1 = __create_conv_layer(x, 3, 64, 3, 'conv1_1/fft_conv', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  conv1_2 = __create_conv_layer(conv1_1, 64, 64, 3, 'conv1_2/fft_conv_1', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  pool1 = tf.layers.max_pooling2d(conv1_2, [2,2], [2,2], padding='valid')
  print('pool1: ', pool1.shape)

  conv2_1 = __create_conv_layer(pool1, 64, 128, 3, 'conv2_1/fft_conv_2', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  conv2_2 = __create_conv_layer(conv2_1, 128, 128, 3, 'conv2_2/fft_conv_3', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  tf.summary.histogram("feature_conv2_2", conv2_2)
  # conv2_1 = __create_conv_layer(pool1, 64, 128, 3, 'conv2_1', isTrain=isTrain)
  # conv2_2 = __create_conv_layer(conv2_1, 128, 128, 3, 'conv2_2', isTrain=isTrain)
  pool2 = tf.layers.max_pooling2d(conv2_2, [2,2], [2,2], padding='valid')
  print('pool2: ', pool2.shape)

  conv3_1 = __create_conv_layer(pool2, 128, 256, 3, 'conv3_1/fft_conv_4', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  conv3_2 = __create_conv_layer(conv3_1, 256, 256, 3, 'conv3_2/fft_conv_5', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  conv3_3 = __create_conv_layer(conv3_2, 256, 256, 3, 'conv3_3/fft_conv_6', isTrain=isTrain, isADMM=opt.admm, isFFT=opt.ffttrain, Budget=self.budget[0])
  tf.summary.histogram("feature_conv3_3", conv3_3)
  # conv3_1 = __create_conv_layer(pool2, 128, 256, 3, 'conv3_1', isTrain=isTrain)
  # conv3_2 = __create_conv_layer(conv3_1, 256, 256, 3, 'conv3_2', isTrain=isTrain)
  # conv3_3 = __create_conv_layer(conv3_2, 256, 256, 3, 'conv3_3', isTrain=isTrain)
  pool3 = tf.layers.max_pooling2d(conv3_3, [2,2], [2,2], padding='valid')
  print('pool3: ', pool3.shape)

  conv4_1 = __create_conv_layer(pool3, 256, 512, 3, 'conv4_1', isTrain=isTrain)
  conv4_2 = __create_conv_layer(conv4_1, 512, 512, 3, 'conv4_2', isTrain=isTrain)
  conv4_3 = __create_conv_layer(conv4_2, 512, 512, 3, 'conv4_3', isTrain=isTrain)
  tf.summary.histogram("feature_conv4_3", conv4_3)
  pool4 = tf.layers.max_pooling2d(conv4_3, [2,2], [2,2], padding='valid')
  print('pool4: ', pool4.shape)

  conv5_1 = __create_conv_layer(pool4, 512, 512, 3, 'conv5_1', isTrain=isTrain)
  conv5_2 = __create_conv_layer(conv5_1, 512, 512, 3, 'conv5_2', isTrain=isTrain)
  conv5_3 = __create_conv_layer(conv5_2, 512, 512, 3, 'conv5_3', isTrain=isTrain)
  tf.summary.histogram("feature_conv5_3", conv5_3)
  pool5 = tf.layers.max_pooling2d(conv5_3, [2,2], [2,2], padding='valid')
  print('pool5: ', pool5.shape)


  conv_flatten = tf.layers.flatten(pool5)

  with tf.variable_scope('fc3'):
    ic = 512
    oc = 10
    w_fc3 = __init_weight([ic, oc], opt.restore, 'fc3/dense/kernel:0')
    b_fc3 = __init_bias([oc], opt.restore, 'fc3/dense/bias:0')
    fc3 = tf.layers.dense(conv_flatten, oc, 
          kernel_initializer=tf.constant_initializer(w_fc3),
          bias_initializer=tf.constant_initializer(b_fc3))

  return fc3


