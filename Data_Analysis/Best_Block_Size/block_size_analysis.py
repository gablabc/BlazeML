#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:54:13 2019

@author: gabriel
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm

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

Xdata = np.array(Xdata)            
Mflops = np.array(Mflops)

## Compute optimal  block size
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

# set number of threads values used
nthreads_count = 3
# enter index of benchmark in file
benchindex = 0


# generate a plot of optimal block-size for all ms and Nthr
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
