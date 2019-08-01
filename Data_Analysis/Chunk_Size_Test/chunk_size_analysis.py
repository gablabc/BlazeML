#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:29:56 2019

@author: gabriel
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm

## Latex Font ##
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Palatino'], 'size':15})
rc('text', usetex=True)

#%%
############### importing training data #################
        
benchmark = 'dmatdmatmult'
ms = 2500

### Reading data file ###
filename =  "data_chunk/marvinCS" + str(benchmark) + str(ms) + ".dat"

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
targets = targets.astype(int)

plt.figure(1, figsize=(8, 6))

for i in range(Xdata.shape[0]):
    plt.plot(targets, Mflops[i, :])
    
plt.grid(True, which="both", ls = "-", alpha = 0.5)     
plt.xscale('log')
plt.xlabel('cs')
plt.ylabel('Mflops')
plt.legend(["Nthr = " + str(int(Xdata[i, 0])) for i in range(Xdata.shape[0])])

#%%

plt.savefig( '/home/gabriel/Desktop/HPXML_research_paper/HPXMLpaper/figures/data_analysis/C_chunk_size/' \
                + str(benchmark) + str(ms) + "_with" + str(int(Xdata[0, 3])) + "ite.pdf")
