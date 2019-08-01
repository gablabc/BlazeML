#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:16:50 2019

@author: gabriel
"""

import blackBoxClasses as bb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

############### importing training data #################
        
### Reading data file ###
filename =  "../data/fulldata.dat"

Xdata = []
Mflops = []

with open(filename,"r") as file:
    n_experiments, n_features, n_targets = [int(j) for j in file.readline().split()]
    targets = np.array([float(j) for j in file.readline().split()[n_features:]])
    for i in range(n_experiments):
        line = file.readline().split()
        Xdata.append([float(j) for j in line[0:n_features]])
        Mflops.append([float(j) for j in line[n_features:]])

Xdata = np.array(Xdata)            
Mflops = np.array(Mflops)

Mflops = Mflops[:, 0:-1]
targets = targets[0:-1]
n_targets -= 1

#remove last features
Xdata = Xdata[:, 0:4]
n_features = 4

######## analyze one specific benchmark 

#benchmarks Vec dvecdvecadd dmatdvecmult
benchs = ['dvecdvecadd' ,'dmatdvecmult', 'dmatdmatadd', 'dmatscalarmult', 'dmattdmatadd', 'dmatdmatmult']
sizes = np.array([8, 5, 5, 5, 5, 8])

nthreads_count = 8
benchindex = 0
Xdataben = np.zeros((1, n_features))
Mflopsben = np.zeros((1, n_targets))
n_expben = 8 * sizes[benchindex]

for i in range(nthreads_count):
    cummulative = np.append(np.array([0]), np.cumsum(sizes))
    start = i * (np.sum(sizes)) + cummulative[benchindex]
    indexes = list(range(start, start + sizes[benchindex]))
    print(indexes)
    Xdataben = np.vstack((Xdataben, Xdata[indexes, :]))
    Mflopsben = np.vstack((Mflopsben, Mflops[indexes, :]))
    
Xdataben = Xdataben[1:, :]
Mflopsben = Mflopsben[1:, :]

## find best chunk size
YdataBen = []
maxPerfBen = []

for i in range(Mflopsben.shape[0]):    
    maxPerf = Mflopsben[i][0]
    bestCs = targets[0]
    j = 1
    cs = targets[j]
    while (cs <= Xdataben[i, 3] and j <= n_targets - 1):
        if Mflopsben[i][j] > maxPerf:
            maxPerf = Mflopsben[i][j]
            bestCs = cs
        j += 1
        cs = targets[min(j, n_targets - 1)]
        
    YdataBen = np.append(YdataBen, bestCs)
    maxPerfBen = np.append(maxPerfBen, maxPerf)
    
#%%
# Measure of importance
# Relative variance
relVar = []
for i in range(Mflopsben.shape[0]):
    perfs = Mflopsben[i, :int(min(Xdataben[i, 3], Mflopsben.shape[1]))]
    relVar.append(np.var(perfs) / np.mean(perfs) * 100)
    
# Average Relative sub optimal performance
sizes = np.mean((maxPerfBen.reshape(-1, 1) - Mflopsben) / maxPerfBen.reshape(-1, 1), axis = 1)
#%%
# Neural Network
mlp = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (200, 200, 200, 200))
model = bb.PostTrainingOptimisationModel(mlp, standardized = True, weighted = True)
model.train(Xdataben, Mflopsben, targets)
model.evaluate(Xdataben, Mflopsben, targets)
#%%
# Decision Tree Regression
dtr = DecisionTreeRegressor(max_depth = 12)
model = bb.PostTrainingOptimisationModel(dtr, standardized = False, weighted = True)
model.train(Xdataben, Mflopsben, targets)
model.evaluate(Xdataben, Mflopsben, targets)
model.score(np.zeros(640), np.ones(64), Mflopsben, targets)

#%%
plt.figure(5)
xx, yy = np.meshgrid(np.linspace(np.min(Xdataben[:, 0]), np.max(Xdataben[:, 0]), 100),
                     np.linspace(np.min(Xdataben[:, 1]), np.max(Xdataben[:, 1]), 100))

# here "model" is your model's prediction (classification) function
P, Z, times = model.predict(np.c_[xx.ravel(), yy.ravel(), yy.ravel(), np.round(yy.ravel() / 4096)]) 
#P, Z = model.predict(np.array([xx.ravel(), yy.ravel(), yy.ravel(), np.round(yy.ravel() / 4096)]).T) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm.get_cmap('jet', 10), alpha = 0.5)
plt.clim(1, 10)

#sizes = np.mean((maxPerfBen.reshape(-1, 1) - Mflopsben) / maxPerfBen.reshape(-1, 1), axis = 1)
sizes = np.array(relVar) / 100
plt.scatter(Xdataben[:, 0], Xdataben[:, 1] , c = YdataBen, cmap=cm.get_cmap('jet', 10),\
                                                             s = sizes, edgecolor = 'k')
plt.clim(1, 10)
plt.colorbar()

