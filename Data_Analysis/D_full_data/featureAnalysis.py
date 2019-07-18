#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:47:28 2019

@author: gabriel
"""
# feature analysis and selection
# https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from matplotlib import cm as cm
import pandas as pd

#%%
############### importing training data #################
        
### Reading data file ###
filename =  "data/fulldata.dat"

Xdata = []
Mflops = []
Mflops2 = []
Ydata = []

with open(filename,"r") as file:
    n_experiments, n_features, n_targets = [int(j) for j in file.readline().split()]
    targets = np.array([float(j) for j in file.readline().split()[n_features:]])
    for i in range(n_experiments):
        line = file.readline().split()
        Xdata.append([float(j) for j in line[0:n_features]])
        Mflops.append([float(j) for j in line[n_features:]])
        Mflops2.append([float(j) for j in line[n_features:]])

MflopsCst = np.array(Mflops)

for i in range(n_targets):
    for j in range(n_experiments):
        if targets[i] > np.ceil((Xdata[j][1] / Xdata[j][4])) * np.ceil((Xdata[j][1] / Xdata[j][5])):
            Mflops[j][i] = -1
            Mflops2[j][i] = 10**7

Xdata = np.array(Xdata)            
Mflops = np.array(Mflops)
Mflops2 = np.array(Mflops2)

## remove cs = 20
Mflops = Mflops[:, 0:-1]
Mflops2 = Mflops2[:, 0:-1]
MflopsCst = MflopsCst[:, 0:-1]
n_targets -= 1

MinperExp = np.min(Mflops2, axis = 1).reshape((-1,1))
MaxperExp = np.max(Mflops, axis = 1).reshape((-1,1)) 

Ydata_indexes = np.where(Mflops == MaxperExp)[1]
Ydata = targets[Ydata_indexes] 
YdataReal = np.max(np.column_stack((Ydata, (Xdata[:, 3] + 4 * Xdata[:, 0] - 1) / (4 * Xdata[:, 0]))), axis=1)  
Yworst_indexes = np.where(Mflops == MinperExp)[1]  

#%%

#remove last feature
Xdata = Xdata[:, 0:4]
n_features = 4

# add best chunk size and worst chunk size as features
Xdata = np.column_stack((Xdata, Ydata))

#normalize features
Xdatanorm = (Xdata - np.min(Xdata, axis=0)) / (np.max(Xdata, axis=0) - np.min(Xdata, axis=0))

# covariance matrix
cor = np.corrcoef(Xdatanorm.T)
#%%
# plot matrix
plt.matshow(cor, cmap=cm.get_cmap('jet'))
plt.colorbar()
groups = ['Nthr', 'Ms', 'Mflops', 'Nblocks', 'Best Cs']
plt.xticks(np.arange(len(groups)), groups)
plt.yticks(np.arange(len(groups)), groups)

#%%
#compare spread vs Nthr, B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xdata[:, 0] + np.random.uniform(-0.5, 0.5, n_experiments) \
           ,np.log(Xdata[:, 2]) + np.random.uniform(-0.2, 0.2, n_experiments) \
           , Xdata[:, 3] + np.random.uniform(-2, 2, n_experiments), c = Xdata[:, 4], cmap=cm.get_cmap('jet'))
#ax.set_yscale('log')
ax.set_xlabel("Nthr")
ax.set_ylabel("Mflop")
ax.set_zlabel("Nblocks")

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xdata[:, 0] + np.random.uniform(-0.5, 0.5, n_experiments) \
           ,np.log(Xdata[:, 2] / Xdata[:, 3] ) \
           , Xdata[:, 3] + np.random.uniform(-2, 2, n_experiments), c = Xdata[:, 4], cmap=cm.get_cmap('jet'))
#ax.set_yscale('log')
ax.set_xlabel("Nthr")
ax.set_ylabel("Work Per thread")
ax.set_zlabel("Nblocks")

#%%
#plot histogram
plt.hist(Ydata, bins=20)

#%%
######## analyze one specific benchmark 

#benchmarks Vec dvecdvecadd dmatdvecmult
benchs = ['dvecdvecadd' ,'dmatdvecmult', 'dmatdmatadd', 'dmatscalarmult', 'dmattdmatadd', 'dmatdmatmult']
sizes = np.array([8, 5, 5, 5, 5, 8])

nthreads_count = 8
benchindex = 5
Xdataben = np.zeros((1, n_features + 1))
Mflopsben = np.zeros((1, n_targets))
n_expben = 8 * sizes[benchindex]

for i in range(nthreads_count):
    cummulative = np.append(np.array([0]), np.cumsum(sizes))
    start = i * (np.sum(sizes)) + cummulative[benchindex]
    indexes = list(range(start, start + sizes[benchindex]))
    print(indexes)
    Xdataben = np.vstack((Xdataben, Xdata[indexes, :]))
    Mflopsben = np.vstack((Mflopsben, MflopsCst[indexes, :]))
    
Xdataben = Xdataben[1:, :]
Mflopsben = Mflopsben[1:, :]

#%%
plt.hist(Xdataben[:,-1])

#%%
plt.scatter(Xdataben[:, 1], Xdataben[:, 3])

#%%
#compare spread vs Nthr, B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(Xdataben[:, 0], np.log(Xdataben[:, 2]), Xdataben[:, 3] , c = Xdataben[:, 4], cmap=cm.get_cmap('jet', 11))

p.set_clim(1, 10)
fig.colorbar(p)
ax.set_xlabel("Nthr")
ax.set_ylabel("Mflop")
ax.set_zlabel("Nblocks")

#%%
plt.scatter(Xdataben[:, 1], Xdataben[:, 2] / Xdataben[:, 3])


