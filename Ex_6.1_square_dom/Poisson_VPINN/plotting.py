import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
with open("loss.pkl",'rb') as pfile:
    hist = pkl.load(pfile)

plt.plot(hist,label='loss')
plt.yscale('log')
plt.legend()
plt.savefig('figure/loss.png')    