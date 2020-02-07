import tensorflow as tf
from option.options import opt
from train import train
from utils import augment
if opt.net=='LeNet':
  from models.LeNet import LeNetFFT
elif opt.net=='cifar10Net':
  from models.cifar10VGG import *

class ModelFFT():
  """Build FFT based network model"""
  build_train_op = train.build_train_op
  model_test = train.test
  model_train = train.train
  data_augment = augment.augment
  net_init = cifar10Net_init
  net = cifar10FFT

  def __init__(self):
    self.tf_x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    self.tf_y = tf.placeholder(tf.int32, (None))
    self.one_hot_y = tf.one_hot(self.tf_y, 10)

    if opt.net == 'LeNet':
      self.budget = [64, 64]
      self.logits = LeNetFFT(self.tf_x, self.budget)
      self.dataset = tf.keras.datasets.mnist
    elif opt.net == 'cifar10Net':
      # self.budget = [64, 64, 64, 64]
      self.imagsize = 32
      self.imagpad = 0
      self.insize = 32
      self.inchnl = 3
      self.tf_x = tf.placeholder(tf.float32, (None, 32, 32, 3))
      self.tf_y = tf.placeholder(tf.int32, (None))
      self.phase = tf.placeholder(tf.bool)
      self.one_hot_y = tf.one_hot(self.tf_y, 10)
      self.net_init()
      self.logits = self.net(self.tf_x, True)
      self.dataset = tf.keras.datasets.cifar10

    # Loss
    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
    self.loss = tf.reduce_mean(self.cross_entropy)
    tf.summary.scalar('network loss', self.loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('network accuracy', self.accuracy)
 
    self.sess = tf.Session()

    # Record
    self.runlog = open(opt.logdir+"/trainFFT.log", "w")

    # Save variables
    self.Param = {}
    for v in tf.global_variables():
      print(v.name)
      if "conv" in v.name or "fc" in v.name:
        self.Param[v.name] = v

    self.saver = tf.train.Saver()
