import tensorflow as tf
import numpy as np
from models.modelADMM import Model
from models.modelFFT import ModelFFT
from option.options import opt

# Define model in space domain, do ADMM
if opt.ffttrain == False:
  tfModel = Model()
  tfModel.sess.run(tf.global_variables_initializer())
  if opt.test == True:
    tfModel.model_test(summary=False)
  else:
    tfModel.build_train_op()
    Param = tfModel.model_train()
else:
  # Define model in frequency domain, do post-ADMM fine-tuning
  tfFFTModel = ModelFFT()
  # Only test pre-trained model
  tfFFTModel.sess.run(tf.global_variables_initializer())
  
  tfFFTModel.model_test(summary=False)
  tfFFTModel.build_train_op()
  Param = tfFFTModel.model_train()
