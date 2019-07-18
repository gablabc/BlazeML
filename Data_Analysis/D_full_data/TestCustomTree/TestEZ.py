#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:33:16 2019

@author: gabriel
"""

import CustomTree as ct
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
from math import trunc

### Test Data
Xdata = np.array([[ 0.5 , 0.25, 1, 100],
                 [ 0.75, 0.3 , 1, 100],
                 [ 0.2 , 0.7 , 1, 100],
                 [ 0.8 , 0.7 , 1, 100],
                 [ 0.3 , 0.9 , 1, 100],
                 [ 0.6 , 0.6 , 1, 100],
                 [ 0.4 , 0.5 , 1, 100],
                 [ 0.1 , 0.8 , 1, 100]])

Mflops = np.array([[100, 1000],
                 [200, 1500],
                 [250, 270],
                 [300, 310],
                 [3000 , 1000],
                 [4000 , 3000],
                 [500 , 10],
                 [2000 , 1000]])

targets = np.array([1 ,2])    
bestCs = targets[np.argmax(Mflops, axis = 1)]
bestPerf = np.max(Mflops, axis = 1)
Data = np.column_stack((Xdata ,Mflops, bestPerf, bestCs))


#%%getMSOP(Data)

tree = ct.CustomDecisionTree(10, 2, 0.90, weighted = True)
tree.n_targets = 2
tree.n_features = 4
tree.train(Xdata, Mflops, targets)
tree.printTreeHeader(tree.root, "EZTree.h")
tree.printTree(tree.root)
#tree.evaluate(Xdata, Mflops, targets, printRes = True)
