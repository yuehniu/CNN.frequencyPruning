import tensorflow as tf
from scipy import signal
import numpy as np

a = tf.constant([i for i in range(1024)], shape=[1,32,32,1], dtype=tf.float32)
w = tf.constant([i for i in range(25)], shape=[5,5,1,1], dtype=tf.float32)

b = tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME')

apad = tf.image.pad_to_bounding_box(a, 0, 0, 36, 36)
afft = tf.spectral.fft2d(tf.cast(tf.transpose(apad, perm=[0,3, 1, 2]), tf.complex64))
print("afft shape: ", afft.shape)

wpad = tf.image.pad_to_bounding_box(tf.transpose(w, perm=[3,0,1,2]), 0, 0, 36, 36)
wfft = tf.spectral.fft2d(tf.cast(tf.transpose(wpad, perm=[0,3,1,2]), tf.complex64))
print("wfft shape: ", wfft.shape)

bfft = tf.multiply(afft, wfft)
bifft = tf.real(tf.spectral.ifft2d(bfft))
bifft = tf.transpose(bifft, perm=[0, 2, 3, 1])

sess = tf.Session()
bb,bbifft, bbfft = sess.run([b, bifft, bfft])

anp = np.array([i for i in range(1024)]).reshape((32,32))
wnp = np.array([i for i in range(25)]).reshape((5,5))
bbnp = signal.convolve2d(anp, wnp, boundary='fill', mode='same')

print(bb[0,0,:,0])
print(bbnp[0,:])
print(bbifft[0,2,:,0])
