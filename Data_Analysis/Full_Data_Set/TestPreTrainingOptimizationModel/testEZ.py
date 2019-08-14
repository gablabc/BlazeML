#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:11:35 2019

@author: gabriel
"""


import blackBoxClasses as bb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier

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
print("tree\n")
dtc = DecisionTreeClassifier()
bbmodel = bb.PreTrainingOptimisationModel(dtc, standardized = False, weighted = True)
bbmodel.train(Xdata, Mflops, targets)
print("log reg\n")
lrc = LogisticRegression(solver = 'liblinear')
bbmodellrc = bb.PreTrainingOptimisationModel(lrc, standardized = True, weighted = True)
bbmodellrc.train(Xdata, Mflops, targets, stratification = [])
preds = bbmodellrc.evaluate(Xdata, Mflops, targets)
print("neural net\n")
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (10, 10))
modelmlp = bb.PreTrainingOptimisationModel(mlp, standardized = True, weighted = True)
modelmlp.train(Xdata, Mflops, targets)