#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:11:05 2019

@author: gabriel
"""


import numpy as np
import math

# Generate a array of k subsets of Xdata and Mflops
def cluster_maker(Xdata, Mflops, k):
    delta = int(math.floor(Xdata.shape[0] / k))
    cluster = [[Xdata[i*delta:(i + 1) * delta + int(i==(k - 1)) * Xdata.shape[0],: ], \
                      Mflops[i*delta:(i + 1) * delta+int(i==(k - 1)) * Xdata.shape[0], :]] for i in range(k)]
    return cluster
    
# Merges all subsets except one.
def merge(cluster, index):
    X = cluster[int(index == 0)][0]
    Mf = cluster[int(index == 0)][1]
    for i in range(int(index==0) + 1, len(cluster)):
        if i == index:
            pass
        else:
            X = np.vstack((X, cluster[i][0]))
            Mf = np.vstack((Mf, cluster[i][1]))
            
    return X, Mf


# Compute the MSOP error and prediction time via cross validation
def getCrossValidationError(blackBoxModel, Xdata, Mflops, targets, k,\
                            otherModels = False, printRes = True):
    
    accuracy = []
    relPerf = []
    overheads = []
    cluster = cluster_maker(Xdata, Mflops, k)
    
    #loop over all folds
    for i in range(k):
        accuracy.append([])
        relPerf.append([])
        overheads.append([])
        
        
        # Train on k-1 sets
        Xfold, Mfold = merge(cluster, i)
        blackBoxModel.train(Xfold, Mfold, targets, printRes = printRes)
        
        # Evaluate on the other set
        Acc, overhead = blackBoxModel.evaluate(cluster[i][0], cluster[i][1] \
                                                     , targets, printRes = printRes)
        
        # Add measurements to array
        accuracy[i].append(Acc[0])
        relPerf[i].append(Acc[1])
        overheads[i].append(overhead)
        
        # this will evaluate with other models 
        if otherModels:
            ## Other models ##
            ## Random Guessing
            acc, MSOP = blackBoxModel.score(np.random.randint(np.min(targets),\
                        np.max(targets), cluster[i][1].shape[0]), cluster[i][1],\
                        targets, printRes = printRes)
            
            accuracy[i].append(acc)
            relPerf[i].append(MSOP)
            
            ## Nite / Nthr
            preds = np.array([round(min(targets[-1], cluster[i][0][j, 3] / cluster[i][0][j, 0])) \
                                                          for j in range(cluster[i][1].shape[0])])
        
            acc, MSOP = blackBoxModel.score(preds, cluster[i][1], targets \
                                            ,printRes = printRes)
        
            accuracy[i].append(acc)
            relPerf[i].append(MSOP)            
            
            # f(x) = Cst
            for j in range(targets.shape[0]):
                acc, MSOP = blackBoxModel.score(targets[j] * np.ones(cluster[i][1].shape[0]), \
                                                cluster[i][1], targets, printRes = False)
                accuracy[i].append(acc)
                relPerf[i].append(MSOP)   
    
        
    
    return np.array(accuracy).T, np.array(relPerf).T, np.array(overheads).T
