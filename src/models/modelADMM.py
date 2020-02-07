import re
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from option.options import opt
from train import train
from utils import pruning
from utils import augment
from models.LeNet import *
from models.cifar10VGG import *

class Model():
  """Container of model, forward, train method."""
  if opt.net == 'LeNet':
    net_init = LeNet_init
    net = LeNet
  elif opt.net =='cifar10Net':
    net_init = cifar10Net_init
    net = cifar10FFT

  build_train_op = train.build_admm_train_op if opt.admm else train.build_train_op
  model_test = train.test
  model_train = train.admm_train if opt.admm else train.train
  data_augment = augment.augment
  if opt.admm:
    update_Z_U = train.update_Z_U
    freq_pruning = pruning.freq_pruning

  def __init__(self):
    self.W = []
    # self.U, self.Z are used to feed to the network during ADMM training
    self.U = []
    self.Z = []

    # Build forward
    self.net_init()
    self.logits = self.net(self.tf_x, self.phase)

    # Build ADMM auxiliary variable
    if opt.admm:
      self.FFTSize = opt.fftsize
      self.rho = tf.placeholder(tf.float32)
      for w in tf.global_variables():
        if "kernel_real" in w.name:
          #if self.admm_lyr in w.name:
          #  break
          print(w.name)
          self.W.append(w)
          self.U.append(tf.placeholder(tf.complex64, w.shape))
          self.Z.append(tf.placeholder(tf.complex64, w.shape))
        elif "kernel_imag" in w.name:
          print(w.name)
          self.W.append(w)
          #self.tfhistogram(w)
    # Define loss
    self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                         logits = self.logits, labels = self.tf_y)
    self.loss = tf.reduce_mean(self.cross_entropy)
    tf.summary.scalar('network loss', self.loss)

    # Statistics
    correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.cast(self.tf_y, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('network accuracy', self.accuracy)


    self.sess = tf.Session()
    # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type="curses")

    # Record
    if opt.admm:
      self.runlog = open(opt.logdir+"/trainadmm.log", "w")
    else:
      self.runlog = open(opt.logdir+"/train.log", "w")

    # Save variable
    self.Param = {}
    for v in tf.global_variables():
      print(v.name)
      if "conv" in v.name or "fc" in v.name:
        self.Param[v.name] = v

  def tfhistogram(self, w):
    wtrans = tf.transpose(w, perm=[3,2,0,1])
    padsize = self.FFTSize - wtrans.shape[2]
    paddings = ((0,0),(0,0),(0,padsize),(0,padsize))
    wpad = tf.pad(wtrans, paddings, mode="CONSTANT")
    wfft = tf.spectral.fft2d(tf.cast(wpad, tf.complex64))
    wabs = tf.abs(wfft)
    tf.summary.histogram(w.name, wabs)
    
