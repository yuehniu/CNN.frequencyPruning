import tensorflow as tf
import numpy as np
from option.options import opt
from sklearn.utils import shuffle
from utils.display import progress_bar

def test(self, summary=True):
  """ Test model accuracy
    summary: whether write tf summary into file
  """
  (_, _),(x_test, y_test) = self.dataset.load_data()
  x_test = x_test / 255.0
  if opt.net == 'cifar10Net':
    x_test[:,:,:,0] = (x_test[:,:,:,0] - 0.4914) / 0.2023
    x_test[:,:,:,1] = (x_test[:,:,:,1] - 0.4822) / 0.1994
    x_test[:,:,:,2] = (x_test[:,:,:,2] - 0.4465) / 0.2010
  if opt.net == 'LeNet':
    pad = self.imagpad
    x_test = np.pad(x_test, ((0,0),(pad,pad),(pad,pad)), 'constant')
  num_examples = len(x_test)
  num_examples = 9984
  total_accuracy = 0.0
  total_loss = 0.0
  total_admm_loss = 0.0
  batch_j = 0
  for offset in range(0, num_examples, opt.batchsize):
    batch_x, batch_y = x_test[offset:offset+opt.batchsize], y_test[offset:offset+opt.batchsize]
    batch_x = np.reshape(batch_x, (-1, self.insize, self.insize, self.inchnl))
    batch_y = np.reshape(batch_y, (-1,))
    #batch_x = np.transpose(batch_x, (0,3,1,2))
    if opt.admm:
      feed_dict = {self.tf_x: batch_x, self.tf_y: batch_y, self.rho: opt.rho}
    else:
      feed_dict = {self.tf_x: batch_x, self.tf_y: batch_y}
    if opt.net=='cifar10Net':
      feed_dict[self.phase] = True

    if opt.admm and summary:
      for lyr in range(int(len(self.W)/2)):
        feed_dict[self.Z[lyr]] = np.complex64(self.ADMM_Z[lyr])
        feed_dict[self.U[lyr]] = np.complex64(self.ADMM_U[lyr])
    if summary:
      if opt.admm:
        loss, admm_loss, accuracy,testsummary = self.sess.run([self.loss, self.ADMMloss, self.accuracy, self.mergedsumm], feed_dict=feed_dict)
        total_admm_loss += (admm_loss * len(batch_x)) 
      else:
        loss, accuracy,testsummary = self.sess.run([self.loss, self.accuracy, self.mergedsumm], feed_dict=feed_dict)
      self.testwriter.add_summary(testsummary, self.cur_epoch*(num_examples/opt.batchsize)+batch_j)
    else:
      loss,accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
    total_loss += (loss * len(batch_x)) 
    total_accuracy += (accuracy * len(batch_x))
    batch_j += 1
    cur_examples = batch_j * opt.batchsize
    progress_bar(cur_examples, num_examples,
      "[TEST ] : loss = {:.5f}, ADMM loss = {:.5f}, accuracy = {:.5f}".format(
      total_loss/cur_examples, total_admm_loss/cur_examples, total_accuracy/cur_examples))
  self.avg_accuracy = total_accuracy / num_examples
  self.avg_admm_loss = total_admm_loss / num_examples
  avg_loss = total_loss / num_examples - self.avg_admm_loss
  self.runlog.write("[TEST ] : loss = {:.5f}, ADMM loss = {:.5f}, accuracy = {:.5f}\n".format(avg_loss, self.avg_admm_loss, total_accuracy/num_examples))

def build_train_op(self):
  """ Build train op
  This is a conventional train op, used to get pre-trained model
  """
  if opt.restore and opt.net=='cifar10Net':
    if opt.admm:
      state = np.load(opt.modelfile)
      stat = state['state'].item()
      self.best_accuracy = stat['acc']
      self.last_epoch = stat['epoch']
    elif opt.ffttrain:
      self.best_accuracy = 0.0
      self.last_epoch = 0
      #state = np.load(opt.modelfile)
      #stat = state['state'].item()
  else:
    self.best_accuracy = 0.0
    self.last_epoch = 0

  initial_lr = opt.lr
  decay_rate = opt.decay
  decay_steps = opt.decaystep # 10 epochs
  self.global_step = tf.Variable(0, trainable=False)
  self.learning_rate = tf.train.exponential_decay(initial_lr, 
    self.global_step, decay_steps, decay_rate, staircase=True)
  self.weight_decay = tf.train.exponential_decay(opt.wdecay, 
    self.global_step, decay_steps, 0.5, staircase=True)
  # self.weight_decay = opt.wdecay 
  
  # Define optimizer
  optimizer = tf.contrib.opt.MomentumWOptimizer(self.weight_decay, self.learning_rate, 0.9) 
  # optimizer = tf.train.AdamOptimizer(self.learning_rate) 
  self.train_op = optimizer.minimize(self.loss)

# Update Z and U
def update_Z_U(self):
  #print("Update Z and U")
  j = 0
  for i in range(0, len(self.W), 2):
    wfft_real = self.sess.run(self.W[i])
    wfft_imag = self.sess.run(self.W[i+1])
    wfft = wfft_real + 1j*wfft_imag
    # Update Z
    self.ADMM_Z[j] = wfft + self.ADMM_U[j]
    #print("ID in caller: %x" %(id(self.ADMM_Z[i])))
    zfreq = self.freq_pruning(self.ADMM_Z[j][:], j)
    self.ADMM_Z[j] = zfreq
    
    # Update U 
    self.ADMM_U[j] = self.ADMM_U[j] + wfft - self.ADMM_Z[j]
    j += 1

# Build train op
def build_admm_train_op(self):
  # Define optimizer
  initial_lr = opt.lr
  decay_rate = opt.decay
  decay_steps = opt.decaystep # 10 epochs
  self.global_step = tf.Variable(0, trainable=False)
  self.learning_rate = tf.train.exponential_decay(initial_lr, 
    self.global_step, decay_steps, decay_rate, staircase=True)

  # Add ADMM loss
  self.ADMMloss = 0.0
  self.test_admmloss = np.inf
  j = 0
  for i in range(0, len(self.W), 2):
    wfft_real = self.W[i]
    wfft_imag = self.W[i+1]
    wfft = tf.complex(wfft_real, wfft_imag)
    # gap = wfft - self.Z[i] + self.U[i]
    gap = wfft - self.Z[j] + self.U[j]
    # gap_abs = tf.math.l2_normalize(tf.abs(gap), axis=[2,3])
    gap_abs = tf.abs(gap)
    self.ADMMloss = tf.add(self.ADMMloss, self.rho*tf.nn.l2_loss(gap_abs))
    j += 1
    
  
  tf.summary.scalar("ADMM loss", self.ADMMloss)
  self.loss += self.ADMMloss
  #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
  optimizer = tf.contrib.opt.MomentumWOptimizer(0.0, self.learning_rate, 0.9) 
  self.train_op = optimizer.minimize(self.loss)

def admm_train(self):
  """ Main train function
  This is a conventional train function, used to get pre-trained models.
  """
  print("Train with ADMM...")
  (x_train, y_train),(_,_) = self.dataset.load_data()
  x_train = x_train / 255.0
  x_train[:,:,:,0] = (x_train[:,:,:,0] - 0.4914) / 0.2023
  x_train[:,:,:,1] = (x_train[:,:,:,1] - 0.4822) / 0.1994
  x_train[:,:,:,2] = (x_train[:,:,:,2] - 0.4465) / 0.2010

  if opt.net == 'LeNet':
    pad = self.imagpad
    x_train = np.pad(x_train, ((0,0),(pad,pad),(pad,pad)), 'constant')

  # Initialze tf variables and ADMM axiliary variables
  self.sess.run(tf.global_variables_initializer())
  # self.ADMM_Z and self.ADMM_U are the variables that needs to to updated
  self.ADMM_Z = []
  self.ADMM_U = []
  for i in range(0, len(self.W), 2):
    wfft_real = self.sess.run(self.W[i])
    wfft_imag = self.sess.run(self.W[i+1])
    wfft = wfft_real + 1j*wfft_imag
    zfreq= self.freq_pruning(wfft, int(i/2))
    self.ADMM_Z.append(zfreq)

    self.ADMM_U.append(np.zeros_like(zfreq, dtype=np.complex64))
     
  num_examples = len(x_train)
  num_examples = num_examples - (num_examples % opt.batchsize)
  admm_rho = opt.rho;

  # Open summary writer
  self.mergedsumm = tf.summary.merge_all()
  self.trainwriter = tf.summary.FileWriter(opt.logdir+'/trainadmm', self.sess.graph)
  self.testwriter = tf.summary.FileWriter(opt.logdir+'/testadmm')
  logstep = 0;
  self.model_test(summary=False)
  for i in range(opt.epochs):
    self.batch_j = 0
    accuracy_epoch = 0
    admm_loss_epoch = 0
    loss_epoch = 0
    if i % 10 == 0:
      self.update_Z_U()
    #if i % 10 == 0:
    #  admm_rho = admm_rho * 0.8;
    # Shuffle data every epoch
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    for offset in range(0, num_examples, opt.batchsize):
      end = offset + opt.batchsize
      self.batch_j += 1
      batch_x, batch_y = x_train[offset:end], y_train[offset:end]
      batch_x = np.reshape(batch_x, (-1, self.insize, self.insize, self.inchnl))
      batch_x = self.data_augment(batch_x[:], opt.augdev)
      batch_y = np.reshape(batch_y, (-1,))
      
      # Run training
      feed_dict = {self.tf_x: batch_x, self.tf_y: batch_y, self.rho: admm_rho, self.global_step:i}
      if opt.net=='cifar10Net':
        feed_dict[self.phase] = True
      for lyr in range(0, int(len(self.W)/2)):
        feed_dict[self.Z[lyr]] = np.complex64(self.ADMM_Z[lyr])
        feed_dict[self.U[lyr]] = np.complex64(self.ADMM_U[lyr])

      _,train_loss,admm_loss, train_accuracy, lr, trainsummary= \
        self.sess.run([self.train_op, self.loss, self.ADMMloss, self.accuracy, self.learning_rate, self.mergedsumm],
                      feed_dict=feed_dict) 
      logstep += 1
      accuracy_epoch += (train_accuracy * opt.batchsize)
      loss_epoch += (train_loss * opt.batchsize)
      admm_loss_epoch += (admm_loss * opt.batchsize)
      accuracy_avg = accuracy_epoch / (self.batch_j * opt.batchsize)
      loss_avg = loss_epoch / (self.batch_j * opt.batchsize)
      admm_loss_avg = admm_loss_epoch / (self.batch_j * opt.batchsize)
      self.trainwriter.add_summary(trainsummary, logstep)

      progress_bar(end, num_examples, "Epoch [{}/{}]: lr = {:.5f}, loss = {:.5f}, accuracy = {:.5f}"
        .format(i, opt.epochs, lr, loss_avg, accuracy_avg))
      if self.batch_j % opt.printstep == 0:
        self.runlog.write("[TRAIN] Epoch {}, Batch {}: lr = {:.5f}, loss = {:.5f}, ADMM loss={:.5f}, accuracy = {:.5f}\n"
          .format(i+1, self.batch_j, lr, loss_avg-admm_loss_avg, admm_loss_avg, accuracy_avg))
    self.cur_epoch = i
    self.model_test(summary=True)
 
    # Save param
    if self.avg_admm_loss < self.test_admmloss:
      self.test_admmloss = self.avg_admm_loss
      print('Save Param')
      params = self.sess.run(self.Param)
      state = {'params': params, 'epoch': i, 'acc': self.avg_accuracy, 'admm': self.avg_admm_loss}
      np.savez(opt.checkpoint+'/cifar10VGG11admm.npz',state=state)

  # Svae Model
  # self.saver.save(self.sess, opt.checkpoint+'/mnistADMM', write_meta_graph=False)

  self.runlog.close() 
      

def train(self):
  """ Main train function
  This is a conventional train function, used to get pre-trained models.
  """
  (x_train, y_train),(_,_) = self.dataset.load_data()
  x_train = x_train / 255.0
  if opt.net == 'cifar10Net':
    x_train[:,:,:,0] = (x_train[:,:,:,0] - 0.4914) / 0.2023
    x_train[:,:,:,1] = (x_train[:,:,:,1] - 0.4822) / 0.1994
    x_train[:,:,:,2] = (x_train[:,:,:,2] - 0.4465) / 0.2010

  if opt.net == 'LeNet':
    pad = self.imagpad
    x_train = np.pad(x_train, ((0,0),(pad,pad),(pad,pad)), 'constant')

  # Train
  self.sess.run(tf.global_variables_initializer())
  num_examples = int(np.floor(len(x_train)/opt.batchsize) * opt.batchsize)

  self.mergedsumm = tf.summary.merge_all()
  if opt.ffttrain:
    self.trainwriter = tf.summary.FileWriter(opt.logdir+'/trainfft', self.sess.graph)
    self.testwriter = tf.summary.FileWriter(opt.logdir+'/testfft')
  else:
    self.trainwriter = tf.summary.FileWriter(opt.logdir+'/train', self.sess.graph)
    self.testwriter = tf.summary.FileWriter(opt.logdir+'/test')
  logstep = 0;
  
  print()
  print('Train...')
  for i in range(opt.epochs):
    self.batch_j = 0
    accuracy_epoch = 0
    loss_epoch = 0
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    for offset in range(0, num_examples, opt.batchsize):
      end = offset + opt.batchsize
      self.batch_j +=1
      batch_x, batch_y = x_train[offset:end], y_train[offset:end]
      batch_x = np.reshape(batch_x, (-1, self.insize, self.insize, self.inchnl))
      batch_x = self.data_augment(batch_x[:], opt.augdev)
      batch_y = np.reshape(batch_y, (-1,))
      # batch_x = np.transpose(batch_x, (0,3,1,2))
      feed_dict={self.tf_x: batch_x, self.tf_y: batch_y, self.global_step: i}
      if opt.net=='cifar10Net':
        feed_dict[self.phase] = True

      _,logits, train_loss,train_accuracy,trainsummary,lr = self.sess.run(
        [self.train_op, self.logits, self.loss, self.accuracy, self.mergedsumm, self.learning_rate], 
        feed_dict=feed_dict)

      logstep += 1
      accuracy_epoch += (train_accuracy * opt.batchsize)
      loss_epoch += (train_loss * opt.batchsize)
      accuracy_avg = accuracy_epoch / (self.batch_j * opt.batchsize)
      loss_avg = loss_epoch / (self.batch_j * opt.batchsize)
      self.trainwriter.add_summary(trainsummary, logstep)

      progress_bar(end, num_examples, "[TRAIN] Epoch [{}/{}], Batch {}: lr = {:.5f}, loss = {:.5f}, accuracy = {:.5f}"
        .format(i, opt.epochs, self.batch_j, lr, loss_avg, accuracy_avg))
      if self.batch_j % opt.printstep == 0:
        self.runlog.write("[TRAIN] Epoch [{}/{}], Batch {}: lr = {:.5f}, loss = {:.5f}, accuracy = {:.5f}\n"
          .format(i, opt.epochs, self.batch_j, lr, loss_avg, accuracy_avg))

    self.cur_epoch = i
    self.model_test(summary=True)
    if self.avg_accuracy > self.best_accuracy:
      self.best_accuracy = self.avg_accuracy
      print('Save Param')
      params = self.sess.run(self.Param)
      state = {'params': params, 'acc': self.best_accuracy, 'epoch': i+1}
      if opt.ffttrain:
        np.savez(opt.checkpoint+'/cifar10VGG11fft.npz',state=state)
      else:
        np.savez(opt.checkpoint+'/cifar10VGG11.npz',state=state)
  
  # self.saver.save(self.sess, opt.checkpoint+'/'+opt.net+'ref', write_meta_graph=False)
  self.runlog.close() 
