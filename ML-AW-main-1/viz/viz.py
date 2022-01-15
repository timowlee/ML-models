import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d

speed = "v15v25"
dataset = "eszueg"

directory = '.'
positive_list = []
negative_list = []
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        if speed in filename.lower() and dataset in filename.lower() and "good" in filename.lower():
            positive_list.append(os.path.join(directory, filename))
        elif speed in filename.lower() and dataset in filename.lower() and "bad" in filename.lower():
            negative_list.append(os.path.join(directory, filename))
        else:
            continue
    else:
        continue
print(positive_list)
print(negative_list)

pos = np.load(positive_list[0])
neg = np.load(negative_list[0])
n= 20

selection_pos= random.choices(range(1,len(pos)), k=n)
selection_neg = random.choices(range(1,len(neg)), k=n)

neg_show = []
for i in selection_neg:
    a = neg[i]
    sample = np.arange(0, len(a), 2).squeeze()
    lin = interp1d(np.arange(len(a)), a)
    b=lin(sample)
    neg_show.append(a)

figure=plt.figure()
plot=figure.add_subplot(111)
plot.plot(a)
plt.show()

pos_show=[]
for i in selection_pos:
    a = pos[i]
    sample = np.arange(0, len(a), 2).squeeze()
    lin = interp1d(np.arange(len(a)), a)
    b=lin(sample)
    pos_show.append(a)
figure=plt.figure()
plot=figure.add_subplot(111)
plot.plot(a)
plt.show()

plt.figure(0)
for i in range(5):
    for j in range(4):
        ax=plt.subplot2grid((5,4), (i,j))
        loc = i*4+j
        ax.plot(neg_show[loc])
plt.show()

plt.figure(1)
for i in range(5):
    for j in range(4):
        ax=plt.subplot2grid((5,4), (i,j))
        loc = i*4+j
        ax.plot(pos_show[loc])
plt.show()

