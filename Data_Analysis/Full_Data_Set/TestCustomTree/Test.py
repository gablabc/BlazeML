#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:50:23 2019

@author: gabriel
"""


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import CustomTree as ct
import pygraphviz as pgv
from math import trunc

## Latex Font ##
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

## print dot files
def dotTreeRecursive(node, Graph):
    if node.terminal:
        Graph.add_node(node.label, label = "pred : " + str(node.pred) + "\nMSOP : " + str(trunc(node.MSOP * 100) / 100) + " (%)")
    else:
        Graph.add_node(node.label, label = "X[" + str(node.feature) + "] < " + str(node.value))
        Graph.add_edge(node.label, node.left.label)
        dotTreeRecursive(node.left, Graph)
        Graph.add_edge(node.label, node.right.label)
        dotTreeRecursive(node.right, Graph)    
    

def dotTree(node):
    G = pgv.AGraph(strict = False, directed = True)
    dotTreeRecursive(node, G)
    G.write("bob.dot")

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
tree = ct.CustomDecisionTree(8, 1, 0.99, weighted = True)
tree.train(Xdataben, Mflopsben, targets)
tree.evaluate(Xdataben, Mflopsben, targets)
tree.score(np.ones((Mflopsben.shape[0])), Mflopsben, targets)
#dotTree(tree.root)
#tree.score(np.ones((Mflopsben.shape[0])), Mflopsben, targets)
#tree.printTree(tree.root)
#tree.printTreeHeader(tree.root, "basicTree.h")

#%%
fig = plt.figure(5)

# compute meshgrid for plot
x_min, x_max = np.min(Xdataben[:, 0]), np.max(Xdataben[:, 0])
y_min, y_max = np.min(Xdataben[:, 1]), np.max(Xdataben[:, 1])
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# here "model" is your model's prediction (classification) function
Z, times = tree.predict(np.c_[xx.ravel(), yy.ravel()]) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm.get_cmap('jet', 10), alpha = 0.5)
plt.clim(1, 10)

sizes = np.mean((maxPerfBen.reshape(-1, 1) - Mflopsben) / maxPerfBen.reshape(-1, 1), axis = 1)
plt.scatter(Xdataben[:, 0], Xdataben[:, 1] , c = YdataBen, cmap=cm.get_cmap('jet', 10),\
                                                             s = 200 * sizes, edgecolor = 'k')
plt.clim(1, 10)
plt.colorbar()
plt.xlabel("Nthr")
plt.ylabel("Vector Size")
plt.title("Predicted chunk size by Custom Decision Tree")
plt.yscale("log")
plt.text(0.75, 0.9, "chunk-size", transform = fig.transFigure)


