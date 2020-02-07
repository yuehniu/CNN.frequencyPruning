"""Visualize loss and accuracy curve"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pattern_loss = re.compile("loss")
pattern_ADMMloss = re.compile("ADMM loss")
pattern_accuracy = re.compile("accuracy")
pattern_num = re.compile("[0-9]*\.[0-9]*")

log_name = sys.argv[1]
Loss = []
Accuracy = []
with open(log_name) as log_handle:
  line = log_handle.readline()
  while(line):
    if "TEST" in line:
      match_loss = pattern_loss.search(line)
      match_string = line[match_loss.end():]
      loss_i = pattern_num.search(match_string) 
      Loss.append(float(match_string[loss_i.start():loss_i.end()]))
      match_accuracy = pattern_accuracy.search(line)
      match_string = line[match_accuracy.end():]
      accuracy_i = pattern_num.search(match_string)
      Accuracy.append(float(match_string[accuracy_i.start():accuracy_i.end()]))
    line = log_handle.readline()


plt.figure()
plt.subplot(211)
plt.plot(Loss, lw=2)
#plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ADMM loss")
plt.subplot(212)
plt.plot(range(40,100),Loss[-60:], lw=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.title("ADMM loss from epoch 100 to 200")

plt.savefig("Loss.png")

plt.figure()
plt.plot(Accuracy, lw=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.savefig("Accuracy.png")
