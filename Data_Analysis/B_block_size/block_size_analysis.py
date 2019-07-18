#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:54:13 2019

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
filename =  "data_block/marvinBlockSizeVec.dat"

Xdata = []
Mflops = []
Ydata = []

with open(filename,"r") as file:
    n_experiments, n_features, n_targets = [int(j) for j in file.readline().split()]
    targets = np.array([j for j in file.readline().split()[n_features:]])
    for i in range(n_experiments):
        line = file.readline().split()
        Xdata.append([float(j) for j in line[0:n_features]])
        Mflops.append([float(j) for j in line[n_features:]])

#for i in range(n_targets):
#    for j in range(n_experiments):
#        if targets[i] > np.ceil((Xdata[j][1] / Xdata[j][4])) * np.ceil((Xdata[j][1] / Xdata[j][5])):
#            Mflops[j][i] = -1

Xdata = np.array(Xdata)            
Mflops = np.array(Mflops)

## Compute optimal 

MaxperExp = np.max(Mflops, axis = 1).reshape((-1,1)) 

Ydata_indexes = np.where(Mflops == MaxperExp)[1]
Ydata = targets[Ydata_indexes] 

#%%
#benchmarks MatADD dmatdmatadd tdmattdmatadd dmattdmatadd tdmatdmatadd
#benchs = ['dmatdmatadd', 'tdmattdmatadd', 'dmattdmatadd', 'tdmatdmatadd']
#sizes = np.array([5, 5, 5, 5])

#benchmarks MatMULT dmatdmatmult tdmattdmatmult dmattdmatmult
#benchs = ['dmatdmatmult', 'tdmattdmatmult', 'dmattdmatmult']
#sizes = np.array([8, 8, 8])

#benchmarks Vec dvecdvecadd dmatdvecmult
benchs = ['dvecdvecadd' ,'dmatdvecmult']
sizes = np.array([8, 5])

#benchmarks dmatdvecmult
#benchs = ['dmatdvecmult']
#sizes = np.array([5])


nthreads_count = 3
benchindex = 0

for i in range(nthreads_count):
    cummulative = np.append(np.array([0]), np.cumsum(sizes))
    start = i * (np.sum(sizes)) + cummulative[benchindex]
    indexes = list(range(start, start + sizes[benchindex]))
    print(indexes)
    color = Ydata_indexes[indexes]
    plt.scatter(Xdata[indexes, 0], Xdata[indexes, 1], c = color, cmap=cm.get_cmap('jet'))
    plt.clim(0,3)

plt.yscale('log')
plt.colorbar()
plt.title(benchs[benchindex])
plt.xlabel('Nthr')
plt.ylabel('Matrix size')

#%%
# plot matrix
plt.matshow(cor, cmap=cm.get_cmap('jet'))
plt.colorbar()
groups = ['Nthr', 'Ms', 'Mflops', 'Nblocks', 't1', 't2', 'd1', 'Best Cs', 'Worst Cs']
plt.xticks(np.arange(len(groups)), groups)
plt.yticks(np.arange(len(groups)), groups)

#%%
df = pd.DataFrame({'Nthr' : Xdata[:, 0], 'Ms' : np.log(Xdata[:,1]), 'Mflop' : np.log(Xdata[:, 2]), 'Nblocks' : Xdata[:, 3], \
                    'Best Cs' : Xdata[:, 7]})
pd.plotting.scatter_matrix(df, figsize=(12, 12))

#%%
# compare cs and Nblocks / Nthr
equalShare = Xdata[:, 3] #/ Xdata[:, 0]
color = (MflopsCst[list(range(n_experiments)), Ydata_indexes] - MflopsCst[list(range(n_experiments)), Yworst_indexes]) / MflopsCst[list(range(n_experiments)), Ydata_indexes] * 100
plt.scatter(equalShare + np.random.uniform(-5, 5, n_experiments), Ydata_indexes + np.random.uniform(-0.2, 0.2, n_experiments), c = color, cmap=cm.get_cmap('jet') )
#plt.colorbar()

#%%
#compare spred with Nblocks
spread = (MflopsCst[list(range(n_experiments)), Ydata_indexes] - MflopsCst[list(range(n_experiments)), Yworst_indexes]) / MflopsCst[list(range(n_experiments)), Ydata_indexes] * 100

plt.scatter(equalShare + np.random.uniform(-5, 5, n_experiments), spread + np.random.uniform(-1, 1, n_experiments))


#%%
#compare spread vs Nthr, B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xdata[:, 0] + np.random.uniform(-0.5, 0.5, n_experiments) \
           ,np.log(Xdata[:, 2]) + np.random.uniform(-0.2, 0.2, n_experiments) \
           , Xdata[:, 3] + np.random.uniform(-2, 2, n_experiments), c = color, cmap=cm.get_cmap('jet'))
#ax.set_yscale('log')
ax.set_xlabel("Nthr")
ax.set_ylabel("Mflop")
ax.set_zlabel("Nblocks")

#%%
#compare cs and block size
equalShare = Xdata[:, 3] / Xdata[:, 0]
#color=Ydata_indexes
color = (MflopsCst[list(range(n_experiments)), Ydata_indexes] - MflopsCst[list(range(n_experiments)), Yworst_indexes]) / MflopsCst[list(range(n_experiments)), Ydata_indexes] * 100

plt.scatter(Xdata[:, 4] + np.random.uniform(-80, 80, n_experiments), Xdata[:, 5] + np.random.uniform(-80, 80, n_experiments), c = color, cmap=cm.get_cmap('jet') )
plt.colorbar()
#%%
#compare spread vs Nthr, B
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xdata[:, 0] + np.random.uniform(-0.5, 0.5, n_experiments) \
           ,np.log(Xdata[:, 2]) + np.random.uniform(-0.2, 0.2, n_experiments) \
           , Xdata[:, 3] + np.random.uniform(-2, 2, n_experiments), c = Ydata_indexes, cmap=cm.get_cmap('jet'))
#ax.set_yscale('log')
ax.set_xlabel("Nthr")
ax.set_ylabel("Mflop")
ax.set_zlabel("Nblocks")
#%%
# spread vs cs
plt.scatter(Ydata_indexes, (MflopsCst[list(range(n_experiments)), Ydata_indexes] - \
            MflopsCst[list(range(n_experiments)), Yworst_indexes]) / MflopsCst[list(range(n_experiments)), Ydata_indexes] * 100 )

