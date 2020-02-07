#import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( '--model', type=str, default='', help='Net model' )
parser.add_argument( '--lyr', type=str, default='conv1', help='Conv layer to be analyzed' )
parser.add_argument('--fftmodel', action="store_true", help='Model file from FFT finetune')
parser.add_argument('--admmmodel', action="store_true", help='Model file from ADMM finetune')
parser.add_argument('--spacemodel', action="store_true", help='Model file from original space training')
parser.add_argument( '--savepath', type=str, default='./result/kernels/', help='Save plot to' )
parser.add_argument('--fftsize', type=int, default=8, help='FFT size when doing frequency-domain conv.')

opt = parser.parse_args()

f = np.load(opt.model)
state = f['state'].item()
params = state['params']
if opt.fftmodel:
  name = 'FFT'
  conv_wreal = params[opt.lyr+'/kernel_real:0']
  conv_wimag = params[opt.lyr+'/kernel_imag:0']
  wfft = conv_wreal + 1j*conv_wimag
if opt.admmmodel:
  name = 'ADMM'
  conv_wreal = params[opt.lyr+'/kernel_real:0']
  conv_wimag = params[opt.lyr+'/kernel_imag:0']
  wfft = conv_wreal + 1j*conv_wimag
if opt.spacemodel:
  name = 'orig'
  w = params[opt.lyr+'/kernel:0']
  w = np.transpose(w, (3,2,0,1))
  padsize = opt.fftsize - w.shape[2]
  paddings = ((0,0),(0,0),(0,padsize),(0,padsize))
  w = np.pad(w, paddings, "constant")
  wfft = np.fft.fft2(w)
print('wfft shape: ', wfft.shape)
for i in range(1):
  plt.figure()
  plt.title('Frequency-domain kernel values')
  for j in range(1):
    print(np.abs(wfft[i,j,:]))
    wfft_abs_sort = np.sort(np.abs(wfft[i,j,:]), axis=None)
    wfft_abs = np.abs( wfft[ i,j,: ] )
    wfft_binary = np.where( wfft_abs==0, 0, 1 )

    plt.plot(wfft_abs_sort[:-1], lw=2)
    #plt.imshow( wfft_binary, interpolation='none', cmap='gray' )

  plt.savefig(opt.savepath+opt.lyr[0:7]+"_"+str(i)+"_"+str(j)+"_"+name+".png")

#top_k = wfft_abs_sort[-64]
#wfft_abs_crop = np.where(np.abs(wfft[i,0,:])<top_k, 0, np.abs(wfft[i,0,:]))

# This is a test for remote editing
