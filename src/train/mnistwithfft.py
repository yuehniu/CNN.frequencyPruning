import tensorflow as tf
import numpy as np
from modelwithfft import LeNet
from hyperparam import BATCHSIZE

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
# x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

# Setup network input and output
tf_x = tf.placeholder(tf.float32, (None, 28, 28, 1))
tf_y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(tf_y, 10)

# Build neural network
logits = LeNet(tf_x)

# Define loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits = logits, labels = one_hot_y)
loss_mean = tf.reduce_mean(cross_entropy)

# Define optimizer
initial_lr = 0.001
decay_rate = 0.9
decay_steps = 10 # 10 epoches
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_lr, 
  global_step, decay_steps, decay_rate, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_mean)

# Statistics
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test
def evaluate(x_data, y_data):
  #num_example = len(x_data)
  num_example = 1024
  total_accuracy = 0
  total_loss = 0
  sess = tf.get_default_session()
  for offset in range(0, num_example, BATCHSIZE):
    batch_x, batch_y = x_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
    batch_x = np.reshape(batch_x, (-1, 28, 28, 1))
    #batch_x = np.transpose(batch_x, (0,3,1,2))
    loss,accuracy = sess.run([loss_mean, accuracy_op], feed_dict={tf_x: batch_x, tf_y: batch_y})
    total_loss += (loss * len(batch_x)) 
    total_accuracy += (accuracy * len(batch_x))
  return total_loss / num_example, total_accuracy / num_example

# Train
epoch = 100
runlog = open("run_mnistwithfft.log", "w")
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  num_example = len(x_train)
  
  print('Train...')
  print()
  for i in range(epoch):
    batch_j = 0;
    for offset in range(0, num_example, BATCHSIZE):
      end = offset + BATCHSIZE
      batch_j +=1
      batch_x, batch_y = x_train[offset:end], y_train[offset:end]
      batch_x = np.reshape(batch_x, (-1, 28, 28, 1))
      #batch_x = np.transpose(batch_x, (0,3,1,2))
      _,train_loss,train_accuracy, lr = sess.run([train_op,loss_mean, accuracy_op, learning_rate], 
        feed_dict={tf_x: batch_x, tf_y: batch_y, global_step: i})

      if batch_j % 10 == 0:
        print("[TRAIN] Epoch {}, Batch {}: lr = {:.5f}, loss = {:.3f}, accuracy = {:.3f}".format(i+1, batch_j, lr, train_loss, train_accuracy))
        runlog.write("[TRAIN] Epoch {}, Batch {}: lr = {:.5f}, loss = {:.3f}, accuracy = {:.3f}\n".format(i+1, batch_j, lr, train_loss, train_accuracy))

    test_loss, test_accuracy = evaluate(x_test, y_test)
    print("[TEST ] Epoch {}: loss = {:.3f}, accuracy = {:.3f}".format(i+1, test_loss, test_accuracy))
    runlog.write("[TEST ] Epoch {}: loss = {:.3f}, accuracy = {:.3f}\n".format(i+1, test_loss, test_accuracy))
  
  runlog.close() 
