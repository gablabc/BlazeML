#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:16:01 2019

@author: gabriel
"""
import blackBoxClasses as bb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

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

#Xdataben[:, [1, 2]] = np.log(Xdataben[:, [1, 2]])
#%%
# logistic regression
lrc = LogisticRegression(solver = 'liblinear', multi_class='ovr')
model = bb.PreTrainingOptimisationModel(lrc, standardized = True, weighted = True)
model.train(Xdataben, Mflopsben, targets)
preds, overhead = model.predict(Xdataben)
print(overhead)
model.evaluate(Xdataben, Mflopsben, targets)


#%%
# Neural Network
mlp = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (20, 20, 20))
model = bb.PreTrainingOptimisationModel(mlp, standardized = True, weighted = True)
model.train(Xdataben, Mflopsben, targets)
preds, overhead = model.predict(Xdataben)
print(overhead)
model.evaluate(Xdataben, Mflopsben, targets)

#%%
#Decision Tree
dtc = DecisionTreeClassifier(max_depth = 8)
model = bb.PreTrainingOptimisationModel(dtc, standardized = False, weighted = True)
model.train(Xdataben, Mflopsben, targets)
preds, overhead = model.predict(Xdataben)
print(overhead)
model.evaluate(Xdataben, Mflopsben, targets)
####
model.score(np.ones((Mflopsben.shape[0])), Mflopsben, targets)

#%%
plt.figure(1)
xx, yy = np.meshgrid(np.linspace(np.min(Xdataben[:, 0]), np.max(Xdataben[:, 0]), 500),
                     np.linspace(np.min(Xdataben[:, 1]), np.max(Xdataben[:, 1]), 500))

# here "model" is your model's prediction (classification) function
#Z = model.predict(model.standardize(np.c_[xx.ravel(), yy.ravel(), yy.ravel(), yy.ravel() / 4096])) 
Z, overheads = model.predict(np.array([xx.ravel(), yy.ravel(), yy.ravel(), yy.ravel() / 4096]).T) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm.get_cmap('jet', 10), alpha = 0.5)
plt.clim(1, 10)

sizes = np.mean((maxPerfBen.reshape(-1, 1) - Mflopsben) / maxPerfBen.reshape(-1, 1), axis = 1)
plt.scatter(Xdataben[:, 0], Xdataben[:, 1] , c = YdataBen, cmap=cm.get_cmap('jet', 10),\
                                                             s = 200 * sizes, edgecolor = 'k')
plt.clim(1, 10)
plt.colorbar()


