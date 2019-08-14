#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:40:34 2019

@author: galabc
"""

import numpy as np
from math import trunc
import time


# Node of the Decision Tree
class Node(object):
    
    def __init__(self, MSOP, Data, depth):
        # stores critical information
        self.MSOP = MSOP            # MSOP of that node
        self.Data = Data            # Data contained in the node
        self.depth = depth          # Depth of the node

        # default value
        self.label = 0              # label of the node     
        self.feature = 0            # feature of the split
        self.value = 0              # value of the split
        self.pred = 0               # predicted value
        self.left = []              # left child
        self.right = []             # right child
        self.terminal = False       # is it a leaf node?
    
        
        
# Decision Tree CLassifier Based on MSOP
class CustomDecisionTree(object):
    
    def __init__(self, max_depth, min_samples, max_MSOP, weighted = False):
        
        # hyperparameters
        self.max_depth = max_depth       # Maximum depth of the tree
        self.min_samples = min_samples   # Minimum number of samples to split
        self.max_MSOP = max_MSOP         # Maximal MSOP to make a split
        
        # If using weighted MSOP
        self.weighted = weighted
        
        self.root = 0                    # root node of the tree
        self.nNode = 0                   # number of nodes (complexity)
        
        # Data related constants
        self.n_features = 0              # number of features
        self.n_targets = 0               # number of targets
        self.targets = []                # list of all targets CS
        
        # Performance measures done at training time
        self.TrainMSOP = 0
        
        # Relative variance used to weight the MSOP (experimental)
        self.relVar = [0, 0]
        
        # Storage of optimal Cs and Perf to avoid reoptimizing performance
        # index 0 Train
        # index 1 Test
        self.bestCs = [0, 0]         # Store best Chunk Size for Train and Test
        self.bestPerf = [0, 0]       # Store best Perf for Train and Test
        
        #Weighted MSOP
        self.threshold = 0.2         # threshold to remove effect of example
        self.alpha = 0               # experimental value to play with weights
        
    
    # Print the hyperparameters of the tree   
    def printHyperParams(self):
        print("max_depth : %s \n min_samples : %s \n max_MSOP : %s" % \
                      (self.max_depth, self.min_samples ,self.max_MSOP))
        
     

    # Sets the hyperparameters of the tree
    def setHyperParams(self, dictio):
        if not dictio:
            print("empty input")
        else:
            if "max_depth" in dictio:
                print("max_depth changed from %s to %s" % (self.max_depth, dictio["max_depth"]))
                self.max_depth = dictio["max_depth"]
            if "min_samples" in dictio:
                print("min_samples changed from %s to %s" % (self.min_samples, dictio["min_samples"]))
                self.min_samples = dictio["min_samples"]
            if "max_MSOP" in dictio:
                print("max_MSOP changed from %s to %s" % (self.max_MSOP, dictio["max_MSOP"]))
                self.max_MSOP = dictio["max_MSOP"]
    
    
    
    # Compute the MSOP on a given data set
    def getMSOP(self, Data):
        if Data.size == 0:
            return 0
        else:
            # compute the weighted MSOP (experimental)
            if self.weighted:
                weights = Data[:, [-1]]
                MSOP = np.max(np.sum(Data[:, self.n_features:self.n_features + self.n_targets]\
                                    / Data[:,[-3]] * weights, axis = 0))
                return MSOP
            # comute the original MSOP
            else:
                MSOP = np.max(np.mean(Data[:, self.n_features:self.n_features + self.n_targets] \
                                  / Data[:, [-3]], axis = 0))
                # MSOP could be sligthly larger than 1 if we have cases where cs > Nite
                MSOP = min(MSOP, 1)
                return MSOP * Data.shape[0]


    # Split a data set into two using the index and value
    def testSplit(self, index, value, Data):
        indexBool = Data[:,index] < value
        return Data[indexBool], Data[~indexBool]
    
    
    
    # Spliting the data set by finding the optimal split
    # !!! Currently the function returns by copy which is not optimal !!!
    def getSplit(self, Data):
        
        # setting up variables
        # store the best index and split value, the resulting best MSOP
        # and the MSOP's for both children
        # and the best groups for both children
        N = Data.shape[0]
        bestIndex, bestValue, bestMSOP, MSOPright, MSOPleft = 0, 0, 0, 0, 0
        bestMSOPright, bestMSOPleft = 0, 0
        bestGroupLeft, bestGroupRight = np.array([]), np.array([])
        
        
        isSplit = False       #see if there was a split   

        # try every feature to make a split
        for index in range(self.n_features):
            # get all middle values
            middles  = np.diff(np.sort(Data[:, index])) / 2 + np.sort(Data[:, index])[:-1]
            # Try spliting with all values
            for val in middles:
                # split groups accordingly
                DataLeft, DataRight = self.testSplit(index, val, Data)
                MSOPleft = self.getMSOP(DataLeft)
                MSOPright = self.getMSOP(DataRight) 
                
                # get the overall MSOP
                if self.weighted:
                    MSOP = (MSOPleft + MSOPright) / np.sum(Data[:, -1])
                else:
                    MSOP = 1 / N * (MSOPleft + MSOPright)
                
                # Check is MSOP is improved and if the split splits data or not
                if MSOP > bestMSOP and DataLeft.size != 0 and DataRight.size != 0:
                    #Update variables
                    bestIndex, bestValue, bestMSOP = index, val, MSOP
                    bestGroupLeft = DataLeft
                    bestGroupRight = DataRight
                    if self.weighted:
                        # not sure why I put a max in these 2 lines ...
                        bestMSOPright = MSOPright / max(np.sum(bestGroupRight[:, -1]), 1)
                        bestMSOPleft = MSOPleft / max(np.sum(bestGroupLeft[:, -1]), 1)
                    else:
                        bestMSOPright = MSOPright / bestGroupRight.shape[0]
                        bestMSOPleft = MSOPleft / bestGroupLeft.shape[0]
                    isSplit = True
        
        # There was no split (happens when points appears multiple times)
        # In that case the left child node will be empty
        if not isSplit:
            return bestIndex, bestValue, bestMSOPleft, MSOP, bestGroupLeft, Data
        # There was a split
        else:
            return bestIndex, bestValue, bestMSOPleft, bestMSOPright, bestGroupLeft, bestGroupRight
    
    
    # Transform a node into a leaf
    def makeLeaf(self, node):
        node.terminal = True
        if node.Data.shape[0] == 1:
            node.pred = node.Data[0, -2]
        else:
            # This prediction doesn't exclude the cases where cs > Nite
            # However the performance difference is very minimal and if it wasn't so
            # The MSOP would ensure that futher splits are required
            if self.weighted:
                node.pred = self.targets[np.argmax(np.sum(node.Data[:, [-1]] * node.Data[:, self.n_features: \
                            self.n_features + self.n_targets] / node.Data[:, [-3]], axis = 0))]
            else:
                node.pred = self.targets[np.argmax(np.sum(node.Data[:, self.n_features: \
                            self.n_features + self.n_targets] / node.Data[:, [-3]], axis = 0))]
    
    
    
    # Function called recursively to generate a tree
    def splitNode(self, node):
        node.label = self.nNode
        
        
        # Make leaf node when terminal condition is reached
        if node.depth >= self.max_depth or node.Data.shape[0] <= self.min_samples or node.MSOP >= self.max_MSOP:
            self.makeLeaf(node)
            if self.weighted:
                self.TrainMSOP += node.MSOP * np.sum(node.Data[:, -1])
            else:
                self.TrainMSOP += node.MSOP * node.Data.shape[0]
            return
        
        # Otherwise attempt to split
        else:
            bestIndex, bestValue, bestMSOPleft, bestMSOPright, bestGroupLeft, bestGroupRight = \
                                                    self.getSplit(node.Data)
                                                    
            # updating the current node with split information
            node.feature = bestIndex
            node.value = bestValue
            
            # There was no split actually so left child is empty
            if bestGroupLeft.size == 0 and bestGroupRight.size != 0:
                self.makeLeaf(node)
                node.label = self.nNode
                
                if self.weighted:
                    self.TrainMSOP += bestMSOPright * np.sum(bestGroupRight[:, -1])
                else:
                    self.TrainMSOP += bestMSOPright * bestGroupRight.shape[0]
                return
            
            # I don't think this will ever happen !!!
#            # There was no split actually    
#            elif bestGroupLeft.size != 0 and bestGroupRight.size == 0:
#                self.makeLeaf(node)
#                node.label = self.nNode
#                
#                if self.weighted:
#                    self.TrainMSOP += bestMSOPleft * np.sum(bestGroupLeft[:, -1])
#                else:
#                    self.TrainMSOP += bestMSOPleft * bestGroupLeft.shape[0]
#                return
            
            # There was a Split   
            else:
                # empty the data in the node because it won't be useful
                node.Data = []
                self.nNode += 1
                node.left = Node(bestMSOPleft, bestGroupLeft, node.depth + 1)
                self.splitNode(node.left)
                
                self.nNode += 1
                node.right = Node(bestMSOPright, bestGroupRight, node.depth + 1)
                self.splitNode(node.right)
        
               
    # Build a tree
    def buildTree(self, Xdata, Mflops, targets):
        # Set data related constants
        self.n_features = Xdata.shape[1]
        self.n_targets = targets.size
        self.targets = targets
        
        # Set number of nodes to 1
        self.nNode = 1
        
        # compute best Cs and Perf for Training Set
        self.bestCs[0] = np.array([])
        self.bestPerf[0] = np.array([])
        
        for i in range(Mflops.shape[0]):    
            maxPerf = Mflops[i][0]
            maxCs = targets[0]
            j = 1
            cs = targets[j]
            # take into account the fact that cs <= Nite
            while (cs <= Xdata[i, 3] and j <= targets.size - 1):
                if Mflops[i][j] > maxPerf:
                    maxPerf = Mflops[i][j]
                    maxCs = cs
                j += 1
                cs = targets[min(j, targets.size - 1)]
                
            self.bestCs[0] = np.append(self.bestCs[0], maxCs)
            self.bestPerf[0] = np.append(self.bestPerf[0], maxPerf)
            
        # Compute Relative Variances of experiments
        # this is useful to see in which experiments chunk size doesn't
        # significantly affect performance. Relative variance is used
        # because of it's statistical interpretation
        relVar = np.array([])
        if self.weighted:
            for i in range(Mflops.shape[0]):
                perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
                relVar = np.append(relVar, (np.var(perfs) / np.mean(perfs) ) )
                weights = relVar **self.alpha * (relVar >= self.threshold).astype(int)
                
        else:
            weights = np.ones((Xdata.shape[0], 1))
        
        # The data inside the tree is organized into 1 block of height n_experiemnts
        # The columns are organized as follow:
        # |  Features  | Perfs for all cs | best perf among cs | best chunk-size | Weight |
        Data = np.column_stack((Xdata ,Mflops, self.bestPerf[0], self.bestCs[0], weights))
        
        
        # Build the Tree
        self.root = Node(self.getMSOP(Data) / np.sum(Data[:, -1]), Data, 0)
        self.splitNode(self.root)
        
        # overall MSOP computed while building tree so no prediction required
        self.TrainMSOP /= np.sum(Data[:, -1])

    
    # Print the Tree (called recursively)
    def printTree(self, node):
        if node.terminal:
            print('%s[N %s, pred %s,MSOP %s]' % (node.depth*'    ', node.Data.shape[0], node.pred,\
                                                                  np.floor((100 * node.MSOP)) / 100))
        else:
            print('%s[X%d < %.3f]' % ((node.depth*'    ', node.feature, node.value)))
            self.printTree(node.left)
            self.printTree(node.right)
            
            
    def printTreeHeaderRecurssive(self, node, FILE):
        if node.terminal:
            FILE.write('%sreturn %s;\n' % ((node.depth+1)*'    ', node.pred))
        else:
            FILE.write("%sif (featureVector[%s] < %s) {\n" % ((node.depth+1)*'    ', node.feature, node.value))
            self.printTreeHeaderRecurssive(node.left, FILE)
            FILE.write("%s} \n" % ((node.depth + 1)*'    '))
            FILE.write("%selse {\n" % ((node.depth + 1)*'    '))
            self.printTreeHeaderRecurssive(node.right, FILE)
            FILE.write("%s} \n" % ((node.depth + 1)*'    '))
            
    # prints a C++ header file containing the tree
    def printTreeHeader(self, node, filename):
        FILE = open(filename, "w")
        FILE.write("#ifndef TREE_HEADER \n#define TREE_HEADER \n")
        FILE.write("#include <vector>\n\n")    
        FILE.write("template <typename T> \n")
        FILE.write("inline int decisionTree(const std::vector<T>& featureVector) { \n")
        self.printTreeHeaderRecurssive(node, FILE)
        FILE.write("}\n#endif")
        
        
    
    # Train the tree using Training Set
    # printRes -> print results of fits in the console
    # stratification -> don't use it. It's just there for polymorphism
    def train(self, Xdata, Mflops, targets, printRes = True,  stratification = []):
        # Set training accuracies to 0
        self.TrainMSOP = 0
        self.TrainAcc = 0
        
        # Build the tree
        self.buildTree(Xdata ,Mflops, targets)
                
        if printRes:
            print("Training done with:\n MSOP " + str(100 * self.TrainMSOP) + " (%)")
        


    # Function called recursively to get prediction of a single point    
    def predictOne(self, node, Xdata):
        if node.terminal:
            return node.pred
        else:
            if Xdata[node.feature] < node.value:
                return self.predictOne(node.left, Xdata)
            else:
                return self.predictOne(node.right, Xdata)
            
            
            
    # Function called to get predictions on an array
    # also returns prediction times in micro-seconds
    def predict(self, Xdata):
        preds = []
        ## Measure Prediction Overhead
        beg = time.time()
        for row in Xdata:
            preds.append(self.predictOne(self.root, row))
        end = time.time()
        return np.array(preds), (end - beg) * 10 ** 9
    
   
    # Returns the score of given predictions
    # It assumes that evaluate() was called once to compute the 
    # bestCS and Perfs on Test set which are stored as attributes of the tree
    # It also assume that the relVar where computed
    # This avoids needing to optimize everytime you want to get a score
    def score(self, preds, Mflops, targets, printRes = True):
        Acc = 100 * np.mean(preds == self.bestCs[1])
        
        if self.weighted:
            weights = (self.relVar[1] > self.threshold).astype(int) * self.relVar[1] ** self.alpha
            MSOP = 100 * np.sum(Mflops[list(range(Mflops.shape[0])), preds.astype(int) - 1] * \
                                       weights / self.bestPerf[1]) / np.sum(weights)
            
        else:
            MSOP = 100 * np.mean(Mflops[list(range(Mflops.shape[0])), preds.astype(int) - 1] / self.bestPerf[1])
        
        
        if printRes:
            print("Test results are in \n Accuracy " + str(trunc(Acc * 100) / 100) + " (%) \n" \
                                                         + "MSOP " + str(trunc(MSOP * 100) / 100) + "\n")
        return Acc, MSOP
    
    
    # Function thet compute the bestCs and bestPerf on Test Set, get predictions and
    # calls the score function. 
    def evaluate(self, Xdata, Mflops, targets, printRes = True):
        # compute best Cs and Perf on Test Set
        self.bestCs[1] = np.array([])
        self.bestPerf[1] = np.array([])
        
        for i in range(Mflops.shape[0]):    
            maxPerf = Mflops[i][0]
            maxCs = targets[0]
            j = 1
            cs = targets[j]
            # must ensure that cs <= Nite
            while (cs <= Xdata[i, 3] and j <= targets.size- 1):
                if Mflops[i][j] > maxPerf:
                    maxPerf = Mflops[i][j]
                    maxCs = cs
                j += 1
                cs = targets[min(j, targets.size - 1)]
                
            self.bestCs[1] = np.append(self.bestCs[1], maxCs)
            self.bestPerf[1] = np.append(self.bestPerf[1], maxPerf)
            
        ## Compute Relative Variances of experiments
        self.relVar[1] = np.array([])
        for i in range(Mflops.shape[0]):
            perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
            self.relVar[1] = np.append(self.relVar[1], (np.var(perfs) / np.mean(perfs) ) )
        
        # Get prediction on Test Set
        preds, overhead = self.predict(Xdata)
        # Return the score
        return self.score(preds, Mflops, targets, printRes = printRes), overhead
        