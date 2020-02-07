# Data augmentation
import numpy as np

def augment(self, x, stddev):
  """ Data augment
    * Random flip
    * Add noise
  """
  datalen = x.shape[0]  
  
  # Flip
  flip = np.random.randint(2, size=datalen)
  for i in range(datalen):
    if flip[i]==1:
      x[i,:] = np.fliplr(x[i,:])

  # Add noise
  # noise = np.random.normal(0, stddev, x.shape)
  return x
