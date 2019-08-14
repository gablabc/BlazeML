#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:33:16 2019

@author: gabriel
"""

import CustomTree as ct
import numpy as np
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import blackBoxClasses as bb


## Latex Font ##
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


### Fake Data
Xdata = np.array([[0, 0, 10, 10], [1, 0, 10, 10], [2, 0, 10, 10], [3, 0, 10, 10],
                  [0, 1, 10, 10], [1, 1, 10, 10], [2, 1, 10, 10], [3, 1, 10, 10],
                  [0, 2, 10, 10], [1, 2, 10, 10], [2, 2, 10, 10], [3, 2, 10, 10],
                  [0, 3, 10, 10], [1, 3, 10, 10], [2, 3, 10, 10], [3, 3, 10, 10]])
                  
                  
Mflops = np.array([[100, 1000], [210, 200], [310, 300], [200, 90], 
                   [100, 1000], [1500, 4000], [3000, 5000], [220, 190], 
                   [100, 110], [100, 450], [1000, 239], [220, 190], 
                   [100, 200], [300, 170], [300, 305], [98, 100]])
                   
                   

targets = np.array([1 ,2])    
bestCs = targets[np.argmax(Mflops, axis = 1)]
bestPerf = np.max(Mflops, axis = 1)
Data = np.column_stack((Xdata ,Mflops, bestPerf, bestCs))

#%%getMSOP(Data)

tree = ct.CustomDecisionTree(10, 1, 0.95, weighted = True)
tree.n_targets = 2
tree.n_features = 4
tree.train(Xdata, Mflops, targets)
tree.printTree(tree.root)

#%%

fig = plt.figure(1, figsize=(10, 5))

# compute meshgrid for plot
x_min, x_max = np.min(Xdata[:, 0]), np.max(Xdata[:, 0])
y_min, y_max = np.min(Xdata[:, 1]), np.max(Xdata[:, 1])
xx, yy = np.meshgrid(np.linspace(x_min-0.25, x_max+0.25, 100),
                     np.linspace(y_min-0.25, y_max+0.25, 100))

sizes = np.mean((bestPerf.reshape(-1, 1) - Mflops) / bestPerf.reshape(-1, 1), axis = 1)

### first plot
ax1 = fig.add_subplot(1, 2, 1)

# here "model" is your model's prediction (classification) function
Z, times = tree.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)


#have 1 black point
Z[0, 0] = 2
ax1.contourf(xx, yy, Z, cmap=cm.get_cmap('Greys'), alpha = 0.3)

# white plot
ax1.scatter(Xdata[bestCs==1, 0], Xdata[bestCs==1, 1] , c = 'white', label = '$cs^*=1$',
                                            s = 1.5*10**3 * sizes[bestCs==1], edgecolor = 'k')
# dark plot
ax1.scatter(Xdata[bestCs==2, 0], Xdata[bestCs==2, 1] , c = 'grey', label = '$cs^*=2$',
                                            s = 1.5*10**3 * sizes[bestCs==2], edgecolor = 'k')

ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_title("PreTO custom decision tree")
fig.legend(loc = (0.41, 0.005), ncol=2)
#%%

#Classical Decision Tree
dtc = DecisionTreeClassifier(max_depth = 10)
model = bb.PreTrainingOptimisationModel(dtc, standardized = False, weighted = True)
strat = sizes
model.train(Xdata, Mflops, targets, stratification = strat)

#%%

### second plot
ax2 = fig.add_subplot(1, 2, 2)

# here "model" is your model's prediction (classification) function
Z, times = model.predict(np.hstack((np.c_[xx.ravel(), yy.ravel()], 10*np.ones((100 * 100, 2))))) 

# Put the result into a color plot
Z = Z.reshape(xx.shape)

#have 1 black point
Z[0, 0] = 2
ax2.contourf(xx, yy, Z, cmap=cm.get_cmap('Greys'), alpha = 0.3)

# white plot
ax2.scatter(Xdata[bestCs==1, 0], Xdata[bestCs==1, 1] , c = 'white', label = 'cs=1',
                                            s = 1.5*10**3 * sizes[bestCs==1], edgecolor = 'k')
# dark plot
ax2.scatter(Xdata[bestCs==2, 0], Xdata[bestCs==2, 1] , c = 'grey', label = 'cs=2',
                                            s = 1.5*10**3 * sizes[bestCs==2], edgecolor = 'k')

ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_title("PreTO classical decision tree")