# Pruning weight in frequency domain
import numpy as np
#TODO 
def freq_pruning(self, Wfreq, Lyr):
  #print("Prune weight in frequency domain")
  shape = Wfreq.shape
  for ochnl in range(shape[0]):
    for ichnl in range(shape[1]):
      w = Wfreq[ochnl,ichnl,:,:]
      w_abs = np.abs(w)
      w_sort = np.sort(w_abs, axis=None)
      top_k = w_sort[-self.budget[0]]
      Wfreq[ochnl,ichnl,:,:] = np.where(w_abs<top_k, 0+0j, w )
  #print("ID in callee, %x" %id(Wfreq))
  return Wfreq
