import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
from option.options import opt

class FFTConv(tf.keras.layers.Layer):
  def __init__(self, FFTSize, TilSize, Wreal, Wimag, B, K, Mode='valid'):
    """ 
    FFTSize: FFT window size for kernel and input
    TilSize: Tile size when decomposing input
    W: Initial spatial weights [h,w,ichnl,ochnl]
    B: Initial spatial bias
    K: Keep K nonzeros values
    """
    super(FFTConv, self).__init__()
    self.fftsize = FFTSize
    self.tilsize = TilSize
    self.krnsize = FFTSize - TilSize + 1
    self.ichnl = Wreal.shape[1]
    self.ochnl = Wreal.shape[0]
    self.binit = B
    self.k = K
    self.mode=Mode
    # padsize = self.fftsize - self.krnsize
    # pads = ((0,0),(0,0),(0,padsize),(0,padsize))
    # wpad = np.pad(np.transpose(self.winit, (3,2,0,1)), pads, "constant")
    # self.wfftinit = np.fft.fft2(wpad)
    self.mask = np.zeros_like(Wreal, dtype=bool)
    print(Wreal.shape)
    if opt.admm:
      self.wrealinit = Wreal
      self.wimaginit = Wimag
    else:
      for oc in range(self.ochnl):
        for ic in range(self.ichnl):
          w = Wreal[oc,ic,:,:] + 1j*Wimag[oc,ic,:,:]
          wabs = np.abs(w)
          wsort = np.sort(wabs, axis=None) 
          topk = wsort[-self.k]
          self.mask[oc, ic, :, :] = wabs >= topk
      if opt.ffttrain: 
        self.wrealinit = np.multiply(Wreal, self.mask)
        self.wimaginit = np.multiply(Wimag, self.mask)
      else:
        self.wrealinit = np.transpose(np.multiply(Wreal, self.mask), (1,0,2,3))
        self.wimaginit = np.transpose(np.multiply(Wimag, self.mask), (1,0,2,3))
        self.mask = np.transpose(self.mask, (1,0,2,3))

  def build(self, DimInput):
    """ Set weights and bias in frequency domain
    DimInput: Input dimension [batch,h,w,ichnl]
    """
    # Setup weight and bias
    # self.wfftinit = tf.cast(self.wfftinit, tf.complex64) 
    self.wreal = tf.Variable(self.wrealinit, name='kernel_real')
    self.wimag = tf.Variable(self.wimaginit, name='kernel_imag')
    if opt.admm:
      self.wfft = tf.complex(self.wreal, self.wimag)
    else:
      self.wfft = tf.multiply(tf.complex(self.wreal, self.wimag), tf.cast(self.mask, tf.complex64))
    self.b = tf.Variable(self.binit, name='bias')

    # Setup number of input tiles
    self.imwidth = DimInput[2].value
    self.imheight = DimInput[1].value 
    self.pad_offset = int((self.krnsize-1)/2)
    self.patchsize = self.fftsize-self.krnsize+1
    self.opatchsize = self.patchsize-(2*self.pad_offset)
    self.cols = int(np.ceil(self.imwidth/self.opatchsize))
    self.rows = int(np.ceil(self.imheight/self.opatchsize))
    self.pkernel = [1,self.patchsize,self.patchsize, 1]
    self.pstride = [1,self.opatchsize, self.opatchsize, 1]
    self.psize = self.rows*self.opatchsize+self.krnsize-1

  # TODO
  def call(self, Input):
    """ Build layer ops
    Input: Input feature map
    """
    Input_padded = tf.image.pad_to_bounding_box(Input, self.pad_offset, self.pad_offset, self.psize, self.psize)
    input_patches = tf.extract_image_patches(Input_padded, self.pkernel, self.pstride, [1,1,1,1], 'VALID')
    patches_space = tf.reshape(input_patches, [-1, self.patchsize, self.patchsize, self.ichnl])
    patches_padded = tf.image.pad_to_bounding_box(patches_space, 0, 0, self.fftsize, self.fftsize)
    patches_to_fft = tf.transpose(patches_padded, perm=[0,3,1,2])
    patches_fft = tf.spectral.fft2d(tf.cast(patches_to_fft, tf.complex64))
    opatches = self.hadamard(patches_fft)
    opatches = tf.transpose(opatches, perm=[0,2,3,1])
    opatches_crop = tf.image.crop_to_bounding_box(opatches, 2*self.pad_offset, 2*self.pad_offset, self.opatchsize, self.opatchsize)
    opatches_depth = tf.reshape(opatches_crop, [-1, self.rows, self.cols, self.opatchsize*self.opatchsize*self.ochnl])
    output = tf.depth_to_space(opatches_depth, self.opatchsize)
    #for r in range(self.rows):
    #  # Row tils
    #  for c in range(self.cols):
    #    # Col tils
    #    # Adjust tile size
    #    if r==(self.rows-1):
    #      irealtilh = self.imheight - (r*self.tilsize)
    #    else:
    #      irealtilh = self.tilsize
    #    if c==(self.cols-1):
    #      irealtilw = self.imwidth - (c*self.tilsize)
    #    else:
    #      irealtilw = self.tilsize

    #    orealtilw = irealtilw + self.krnsize - 1
    #    orealtilh = irealtilh + self.krnsize - 1
    #      
    #    # Extract one tile
    #    # offset = [[r*self.tilsize+irealtilh/2, c*self.tilsize+irealtilw/2] for i in range(opt.batchsize)]
    #    offset = [r*self.tilsize, c*self.tilsize]
    #    # itil = tf.image.extract_glimpse(Input, [irealtilh,irealtilw], offset, centered=False, normalized=False)
    #    itil = tf.image.crop_to_bounding_box(Input, offset[0], offset[1], irealtilh, irealtilw )
    #    # padh = self.fftsize - irealtilh
    #    # padw = self.fftsize - realtilw
    #    itilpads = tf.image.pad_to_bounding_box(itil, 0, 0, self.fftsize, self.fftsize)
    #    
    #    # Convolution in frequency domain
    #    itilpads = tf.transpose(itilpads, perm=[0, 3, 1, 2]) #[batch,h,w,ichannel]->[batch,ichannel,h,w]
    #    itilfft = tf.spectral.fft2d(tf.cast(itilpads, tf.complex64))
    #    otil = self.hadamard(itilfft)
    #    otil = tf.transpose(otil, perm=[1,2,3,0]) #[ochnl,batch,h,w]-->[batch,h,w,ochnl]
    #     
    #    otilcrop = tf.image.crop_to_bounding_box(otil, 0, 0, orealtilh, orealtilw)
    #    # Concat tiles along cols
    #    if 0 == c:
    #      ocols = otilcrop
    #    else:
    #      overlapcols = self.krnsize - 1
    #      overlap =  ocols[:,:,-overlapcols:,:] + otilcrop[:,:,0:overlapcols,:] 
    #      ocols = tf.concat([ocols[:,:,0:-overlapcols,:], overlap, otilcrop[:,:,overlapcols:,:]], 2)
    #  # Concat tiles among rows
    #  if 0 == r:
    #    orows = ocols
    #  else:
    #    overlaprows = self.krnsize - 1
    #    overlap = orows[:,-overlaprows:,:,:] + ocols[:,0:overlaprows,:,:]
    #    orows = tf.concat([orows[:,0:-overlaprows,:,:], overlap, ocols[:,overlaprows:,:,:]], 1)

    # output = tf.transpose(output, perm=[0,2,3,1])
    bias = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.b, 0), 0),0)
    #bias = tf.tile(bias, [opt.batchsize, 1, 1, 1])
    output = output + bias
    if self.mode=='valid':
      crops = self.krnsize-1;
      owidth = self.imwidth - (self.krnsize-1)
      oheight = self.imheight - (self.krnsize-1)
    elif self.mode=='same':
      crops = 0
      owidth = self.imwidth
      oheight = self.imheight
    return tf.image.crop_to_bounding_box(output, crops, crops, oheight, owidth)

  def hadamard(self, ITil):
    """ Modified Hadamard product
    ITil: Frequency-domain input
    """
    wfft_trans = tf.transpose(self.wfft, perm=[2,3,0,1])
    itil_trans = tf.transpose(ITil, perm=[2,3,1,0])
    otils = tf.matmul(wfft_trans, itil_trans)
    otils_trans = tf.transpose(otils, perm=[3,2,0,1])
    otils_ifft = tf.spectral.ifft2d(otils_trans)
    otils_batch = tf.real(otils_ifft)
    #wfft_expand = tf.expand_dims(self.wfft, 1)
    #wfft_batch = tf.tile(wfft_expand,[1, tf.shape(ITil)[0], 1, 1, 1])
    #otils = tf.multiply(wfft_batch, ITil)
    #otilbatch = tf.spectral.ifft2d(tf.reduce_sum(otils, 2))
    #otilbatch = tf.real(otilbatch)
    # otilfft0 = tf.multiply(self.wfft, ITil[0,:])
    # otilbatch0 = tf.spectral.ifft2d(tf.reduce_sum(otilfft0, 1))
    # #otilbatch0 = tf.reduce_sum(tf.spectral.ifft2d(otilfft0), 1)
    # otilbatch = tf.expand_dims(tf.real(otilbatch0), 0) 
    # for b in range(1, opt.batchsize):
    #   otilffti = tf.multiply(self.wfft, ITil[b,:]) 
    #   otilbatchi = tf.spectral.ifft2d(tf.reduce_sum(otilffti, 1))
    #   #otilbatchi = tf.reduce_sum(tf.spectral.ifft2d(otilffti), 1)
    #   otilbatch = tf.concat([otilbatch, tf.expand_dims(tf.real(otilbatchi), 0)], 0)

    return otils_batch
