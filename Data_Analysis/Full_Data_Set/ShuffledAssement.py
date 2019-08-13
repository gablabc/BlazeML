#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:59:29 2019

@author: gabriel
"""


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import CustomTree as ct
from matplotlib import cm as cm
import blackBoxClasses as bb
import crossvalid as cv


############### importing training data #################
        
### Reading data file ###
filename =  "data/fulldata.dat"

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

#remove last features
Xdata = Xdata[:, 0:4]
n_features = 4

# Random Permutation of the Data

perms = np.random.permutation(n_experiments)
P = np.zeros((n_experiments, n_experiments))

for i in range(n_experiments):
    for j in range(n_experiments):
        P[i][perms[i]] = 1

P = P.astype(int)

Xdata = np.dot(P, Xdata)
Mflops = np.dot(P, Mflops)


# Train Test Split  (1 : 2)

XTrain = Xdata[0:192, :]
MflopsTrain = Mflops[0:192, :]

XTest = Xdata[192:, :]
MflopsTest = Mflops[192:, :]

#%%

## Custom Model ##

customBlackBoxcdt = ct.CustomDecisionTree(20, 1, 0.95, weighted = True)



## Pre Training Optimisation Models

# Decision Tree Classifier
dtc = DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 4)
preBlackBoxdtc = bb.PreTrainingOptimisationModel(dtc, standardized = False, weighted = True)

# Logistic Regression
lrc = LogisticRegression(solver = 'liblinear', multi_class='ovr')
preBlackBoxlrc = bb.PreTrainingOptimisationModel(lrc, standardized = True, weighted = True)



## Post Training Optimisation Model
dtr2 = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 4)
postBlackBoxdtr = bb.PostTrainingOptimisationModel(dtr2, standardized = False, weighted = True)

mlp2 = MLPRegressor(hidden_layer_sizes = (100, 100, 100))
postBlackBoxmlp = bb.PostTrainingOptimisationModel(mlp2, standardized = True, weighted = True)

#%%
# Hyper-parameter search for Custom Tree
customBlackBoxcdt.setHyperParams({"max_depth" : 10, "min_samples" : 1, "max_MSOP": 0.99})
accuracy, relPerf, overheads = cv.getCrossValidationError(customBlackBoxcdt, XTrain, MflopsTrain, targets, 3\
                                               , printRes = False, otherModels = False)
print(str(np.mean(relPerf)) + " (%) \n")
print(str(np.mean(overheads)))
#customBlackBoxcdt.printTree(customBlackBoxcdt.root)
#%%

# Hyper-parameter search for Classification Tree
for depth in [20, 18, 16, 14, 12, 10, 8 ,6, 4]:
    preBlackBoxdtc.setHyperParams({"max_depth" : depth, "min_samples_leaf" : 1})
    accuracy, relPerf , overheads = cv.getCrossValidationError(preBlackBoxdtc, XTrain, MflopsTrain, targets, 3,\
                                                              printRes = False, otherModels = False)
    print(str(depth) + " : " + str(np.mean(relPerf)) + " (%) \n")
     
#%%
# Set hyperparams
preBlackBoxdtc.setHyperParams({"max_depth" : 10, "min_samples_leaf" : 1})
accuracy, relPerf, overheads = cv.getCrossValidationError(preBlackBoxdtc, XTrain, MflopsTrain, targets, 3\
                                                          ,printRes = True, otherModels = False)
print(str(depth) + " : " + str(np.mean(relPerf)) + " (%) \n")
    
#%%   
#preBlackBoxlrc.setHyperParams()
accuracy, relPerf, overheads = cv.getCrossValidationError(preBlackBoxlrc, XTrain, MflopsTrain, targets, 3, \
                                               printRes = True, otherModels = False)

#%%
#Hyper Parameter search for mlp post training optimisation
#postBlackBoxmlp

postBlackBoxmlp.setHyperParams({"hidden_layer_sizes" : (100, 100, 100), "solver" : 'lbfgs'})
accuracy, relPerf, overheads = cv.getCrossValidationError(postBlackBoxmlp, XTrain, MflopsTrain, targets, 3\
                                               , printRes = False, otherModels = False)
print(str(np.mean(relPerf)) + " (%) \n")
print(str(np.mean(overheads)))

#%%
postBlackBoxdtr.setHyperParams({"max_depth" : 20, "min_samples_leaf" : 1})
accuracy, relPerf, overheads = cv.getCrossValidationError(postBlackBoxdtr, XTrain, MflopsTrain, targets, 3\
                                               , printRes = False, otherModels = False)
print(str(np.mean(relPerf)) + " (%) \n")
print(str(np.mean(overheads)))

#%%

# Train Selected model on Training Set and Test on Test Set
#blackBoxdtr.setHyperParams({"max_depth" : 6})
customBlackBoxcdt.train(XTrain, MflopsTrain, targets)
customBlackBoxcdt.evaluate(XTest, MflopsTest, targets)
customBlackBoxcdt.printTree(customBlackBoxcdt.root)
customBlackBoxcdt.printTreeHeader(customBlackBoxcdt.root, "TrainingSetTree.h")

#%%
import pygraphviz as pgv
from math import trunc

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
    G.write("FullDataTree.dot")

#dotTree(customBlackBoxcdt.root)

#%%
    
# generate the header file of the fitted decision tree
customBlackBoxcdt.train(Xdata, Mflops, targets)
customBlackBoxcdt.evaluate(Xdata, Mflops, targets)
customBlackBoxcdt.printTree(customBlackBoxcdt.root)
customBlackBoxcdt.printTreeHeader(customBlackBoxcdt.root, "FullSetTree.h")