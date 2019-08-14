#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:36:08 2019

@author: gabriel
"""

import numpy as np
import time

class PreTrainingOptimisationModel(object):
    
    # Initiate all member attributes
    def __init__(self, algorithm, standardized, weighted):
        
        self.alg = algorithm               # Intrinsic ML algorithm
        self.standardized = standardized   # Standardizing inputs
        self.weighted = weighted           # Weights when assessing Perf
        
        
        self.bestCs = [0, 0]        # Store best Chunk Size for Train and Test
        self.bestPerf = [0, 0]      # Store best Perf for Train and Test
        
        # Values for standardization
        self.mean = []
        self.std = []
        
        # relative Variances for train and test set (Importance of classif)
        self.relVar = [0, 0]
        
        # Data related constants
        self.targets = []
        self.n_features = 0
        self.n_targets = 0
        
        # Weighted MSOP
        self.threshold = 0.2
        self.alpha = 0
        
        
    # see the hyperparameters of the underlying algorithm    
    def printHyperParams(self):
        print(self.alg.get_params())
        
        
    def setHyperParams(self, paramsDict):
        if type(paramsDict) == dict:
            self.alg.set_params(**paramsDict)
            print("Hyperparameters were changed")
        else:
            print("You must input a dictionary")
    
    
    def standardize(self, Xdata):
        return (Xdata - self.mean) / self.std
    
    
    
    # Pre-Training optimisation
    # Sets the values of attributes inside the black box model
    # Index = 0 Training
    # Index = 1 Test
    # IMPORTANT  Standardization must not be called before this
    def optimize(self, Xdata, Mflops, index):
        # Reset the values
        self.bestCs[index] = []
        self.bestPerf[index] = []
        
        # Compute the best Cs and Perfs while ensuring cs <= Nite
        for i in range(Xdata.shape[0]):    
            maxPerf = Mflops[i][0]
            maxCs = self.targets[0]
            j = 1
            cs = self.targets[j]
            while (cs <= Xdata[i, 3] and j <= self.n_targets - 1):
                if Mflops[i][j] > maxPerf:
                    maxPerf = Mflops[i][j]
                    maxCs = cs
                j += 1
                # min is required so I dont index outside of targets
                cs = self.targets[min(j, self.n_targets - 1)]
                
            self.bestCs[index] = np.append(self.bestCs[index], maxCs)
            self.bestPerf[index] = np.append(self.bestPerf[index], maxPerf)
            
    
    # Training function that calls optimization
    def train(self, Xdata, Mflops, targets, printRes = True, stratification = []):
        # Set constants
        self.n_targets = Mflops.shape[1]
        self.n_features = Xdata.shape[1]
        self.targets = targets
        
        ## Compute Relative Variances of experiments
        self.relVar[0] = np.array([])
        for i in range(Mflops.shape[0]):
            perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
            self.relVar[0] = np.append(self.relVar[0], (np.var(perfs) / np.mean(perfs) ) )
            
        ## Optimisation step before training
        self.optimize(Xdata, Mflops, index = 0)

        ## Standardisation
        if self.standardized:
            self.mean = np.mean(Xdata, axis = 0)
            self.std = np.std(Xdata, axis = 0)
            # Cases where there is no variation
            self.std[self.std == 0] = 1
            Xdata = self.standardize(Xdata)
      
        ## Training Phase
        
        # doing weighted loss function or not
        if stratification == []:
            self.alg.fit(Xdata, self.bestCs[0])
        else:
            self.alg.fit(Xdata, self.bestCs[0], sample_weight = stratification)
        
        ## Output accuracies
        predTrain = np.round(self.alg.predict(Xdata))
        # Ensure that cs <= maxTarget
        predTrain[predTrain > targets[-1]] = targets[-1]
        
        # Compute Accuracies and MSOP
        accuracy = 100 * np.mean(self.bestCs[0] == predTrain)
        
        if self.weighted:
            weights = (self.relVar[0] > self.threshold).astype(int) * self.relVar[0] ** self.alpha
            MSOP = 100 * np.sum(Mflops[list(range(Xdata.shape[0])), predTrain.astype(int) - 1] * \
                                       weights / self.bestPerf[0]) / np.sum(weights)
        else:
            MSOP = 100 * np.mean(Mflops[list(range(Xdata.shape[0])), predTrain.astype(int) - 1] / self.bestPerf[0])
        
        if printRes:
            print("Training done with \n Accuracy : " + str(round(accuracy* 100) / 100) + \
                                                      " (%) \\n MSOP : " + str(round(MSOP * 100) / 100) + " (%)\n")
        
        return accuracy, MSOP
    
                     
    def predict(self, Xdata):
        if (Xdata.shape[1] != self.n_features):
            print("This Data is not compatible with the Training Data")
            
            return []
        else:
            ## begin to count the overhead of prediction at runtime
            beg = time.time()
            
            ## Standardisation
            if self.standardized:
                Xdata = self.standardize(Xdata)
            
            preds = np.round(self.alg.predict(Xdata))
            # ensure that cs < max(targets)
            preds[preds > self.targets[-1]] = self.targets[-1]
            
            ## end count
            end = time.time()
            return preds, (end - beg) * 10 ** 9
    
    
    
    # Assumes that the optimization method has been called via the evaluate function
    # Score is usefull when doing hyperparameter search as the Black Box model doesn't 
    # need to find the optimal chunk sizes every time.
    def score(self, predTest, Mflops, targets, printRes = True):
        n_experimentTest = predTest.shape[0]
        
        accuracy = 100 * np.mean(self.bestCs[1] == predTest)
        if self.weighted:
            weights = (self.relVar[1] > self.threshold).astype(int) * self.relVar[1] ** self.alpha
            MSOP = 100 * np.sum(Mflops[list(range(Mflops.shape[0])), predTest.astype(int) - 1] * \
                                       weights / self.bestPerf[1]) / np.sum(weights)
        else:
            MSOP = 100 * np.mean(Mflops[list(range(n_experimentTest)), predTest.astype(int) - 1] / self.bestPerf[1].T)
        
        if printRes:
            print("Test results are in \n Accuracy : " + str(round(accuracy* 100) / 100) + " (%) \n MSOP : " + \
                                                                              str(round(MSOP * 100) / 100) + " (%)\+n")
        
        return accuracy, MSOP
    
    
    
    # This func
    def evaluate(self, Xdata, Mflops, targets, printRes = True):
        
        if (Xdata.shape[1] != self.n_features or Mflops.shape[1] != self.n_targets):
            print("This Data is not compatible with the Training Data")
            return [], []
        else:
            ## Compute Relative Variances of experiments
            self.relVar[1] = np.array([])
            for i in range(Mflops.shape[0]):
                perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
                self.relVar[1] = np.append(self.relVar[1], (np.var(perfs) / np.mean(perfs) ) )
                
            ## Optimisation step before training and standardization
            self.optimize(Xdata, Mflops, index = 1)
    
            ## Output accuracies
            predTest, overhead = self.predict(Xdata)
            return self.score(predTest, Mflops, printRes), overhead
        
    
            
###############################################################################
            
        
###############################################################################
    
    
    
class PostTrainingOptimisationModel(object):
    
    # Initiate all member attributes
    def __init__(self, algorithm, standardized, weighted):
        
        self.alg = algorithm               # Intrinsic ML algorithm
        self.standardized = standardized   # Standardizing inputs
        self.weighted = weighted           # Weights when assessing Perf        
        
        self.bestCs = [0, 0]         # Store best Chunk Size for Train and Test
        self.bestPerf = [0, 0]      # Store best Perf for Train and Test
        
        # store the Nite for each experiment to ensure that cs <= Nite even
        # after standardization
        self.Nite = [0, 0]
        
        # Values for standardization
        self.mean = []
        self.std = []
        
        # relative Variances for train and test set (Importance of classif)
        self.relVar = [0, 0]
        
        # Data related constants
        self.targets = []
        self.n_features = 0
        self.n_targets = 0
        
        # Weighted MSOP
        self.threshold = 0.2
        self.alpha = 0
        
        
    # see the hyperparameters of the underlying algorithm    
    def printHyperParams(self):
        print(self.alg.get_params())
        
        
    def setHyperParams(self, paramsDict):
        if type(paramsDict) == dict:
            self.alg.set_params(**paramsDict)
            print("Hyperparameters were changed")
        else:
            print("You must input a dictionary") 

    def standardize(self, XdataAug):
        return (XdataAug - self.mean) / self.std
    
    # Pre-Training optimisation
    # Sets the values of attributes inside the black box model
    # Index = 0 Training
    # Index = 1 Test
    # IMPORTANT  Standardization must not be called before this
    def optimize(self, Xdata, Mflops, index):
        # Reset the values
        self.bestCs[index] = []
        self.bestPerf[index] = []
        
        # Compute the best Cs and Perfs while ensuring cs <= Nite
        for i in range(Xdata.shape[0]):    
            maxPerf = Mflops[i][0]
            maxCs = self.targets[0]
            j = 1
            cs = self.targets[j]
            while (cs <= Xdata[i, 3] and j <= self.n_targets - 1):
                if Mflops[i][j] > maxPerf:
                    maxPerf = Mflops[i][j]
                    maxCs = cs
                j += 1
                # min is required so i dont index outside of targets
                cs = self.targets[min(j, self.n_targets - 1)]
                
            self.bestCs[index] = np.append(self.bestCs[index], maxCs)
            self.bestPerf[index] = np.append(self.bestPerf[index], maxPerf)
            
        print("Optimization Done !!!")
        
    # This functions does the feature augmentation procedure where chunk-size
    # is now considered a feature of the Performance Model
    def augmente(self, Xdata):
        chunk_size = [self.targets[0]] * Xdata.shape[0]
        XdataAug = np.column_stack((Xdata, chunk_size))
        for i in range(1, self.n_targets):
            chunk_size = [self.targets[i]] * Xdata.shape[0]
            XdataAug = np.vstack((XdataAug, np.column_stack((Xdata, chunk_size))))
        
        return XdataAug

          
    # Training function that calls optimization
    def train(self, Xdata, Mflops, targets, printRes = True,  stratification = []):
        # Set constants
        self.n_targets = Mflops.shape[1]
        self.n_features = Xdata.shape[1]
        self.targets = targets
        self.Nite[0] = Xdata[:, 3]
        
        ## Compute Relative Variances of experiments
        self.relVar[0] = np.array([])
        for i in range(Mflops.shape[0]):
            perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
            self.relVar[0] = np.append(self.relVar[0], (np.var(perfs) / np.mean(perfs) ) )
        
        ## Optimisation step before training
        self.optimize(Xdata, Mflops, index = 0)
        
        ### Augmente the data set
        XdataAug = self.augmente(Xdata)
        
        ## Standardisation of augmented data
        if self.standardized:
            self.mean = np.mean(XdataAug, axis = 0)
            self.std = np.std(XdataAug, axis = 0)
            # Cases where there is no variation
            self.std[self.std == 0] = 1
            XdataAug = self.standardize(XdataAug)
      
        ## Training Phase
        realPerfs = np.ravel(Mflops, order='F')
        # fit the log of performance to ensure that no example dominates others
        self.alg.fit(XdataAug, np.log(realPerfs))
        
        ## Get Predictions on Training Set
        predPerfs = np.exp(self.alg.predict(XdataAug))
        predCs = self.targets[np.argmax(np.reshape(predPerfs, (Xdata.shape[0], \
                                                self.n_targets), order = 'F'), axis = 1)]
        ## ensure that cs <= Nite
        predCs[predCs >= self.Nite[0]] = self.Nite[0][predCs >= self.Nite[0]]
        
    
        ## Asses performance
        accuracy = 100 * (1 - np.mean(np.abs(predPerfs - realPerfs) / realPerfs))
        if self.weighted:
            weights = (self.relVar[0] > self.threshold).astype(int) * self.relVar[0] ** self.alpha
            MSOP = 100 * np.sum(Mflops[list(range(Xdata.shape[0])), predCs.astype(int) - 1] * \
                                       weights / self.bestPerf[0]) / np.sum(weights)
        else:
            MSOP = 100 * np.mean(Mflops[list(range(Xdata.shape[0])), predCs.astype(int) - 1] / self.bestPerf[0])
        
        if printRes:
            print("Training done with \n Accuracy : " + str(round(accuracy* 100) / 100) + \
                                                      " (%) \\n MSOP : " + str(round(MSOP * 100) / 100) + " (%)\n")
        
        return accuracy, MSOP
    
    
    # This function simply returns the predictions of the model on a new example
    def predict(self, Xdata):        
        if (Xdata.shape[1] != self.n_features):
            print("This Data is not compatible with the Training Data")
            
            return []
        else:
            # set the number of iterations before standardizing
            self.Nite[1] = Xdata[:, 3]
            
            ## Measure Prediction OverHead
            beg = time.time()
            
            ### Augmente the data set
            XdataAug = self.augmente(Xdata)
            ## Standardisation
            if self.standardized:
                XdataAug = self.standardize(XdataAug)
            
            ## Get Predictions of Performance
            predPerfs = np.exp(self.alg.predict(XdataAug))
            ## Optimization with respect to chunk size (overhead)
            predCs = self.targets[np.argmax(np.reshape(predPerfs, (Xdata.shape[0], \
                                                self.n_targets), order = 'F'), axis = 1)]
            ## ensure that cs <= Nite
            predCs[predCs >= self.Nite[1]] = self.Nite[1][predCs >= self.Nite[1]]
            
            # End measuring time
            end = time.time()
            return predPerfs, predCs, (end - beg) * 10 ** 9
     
        
    # Assumes that the optimization method has been called via the evaluate function
    # and that RelVar were computed
    # Here predPerfs is augmented and predCs is not augmented
    def score(self, predPerfs, predCs, Mflops, targets, printRes = True):
        realPerfs = np.ravel(Mflops, order='F')
        
        ## Asses performance
        accuracy = 100 * (1 - np.mean(np.abs(predPerfs - realPerfs) / realPerfs))
        if self.weighted:
            weights = (self.relVar[1] > self.threshold).astype(int) * self.relVar[1] ** self.alpha
            MSOP = 100 * np.sum(Mflops[list(range(Mflops.shape[0])), predCs.astype(int) - 1] * \
                                       weights / self.bestPerf[1]) / np.sum(weights)
        else:
            MSOP = 100 * np.mean(Mflops[list(range(Mflops.shape[0])), predCs.astype(int) - 1] / self.bestPerf[1])
        
        if printRes:
            print("Test results are in \n Accuracy : " + str(round(accuracy* 100) / 100) + " (%) \n MSOP : " + \
                                                                              str(round(MSOP * 100) / 100) + " (%)\+n")
        
        return accuracy, MSOP    
    
    
    # This function will compute relVariances and optimal chunk size on all examples in
    # The given set
    def evaluate(self, Xdata, Mflops, targets, printRes = True):
        
        if (Xdata.shape[1] != self.n_features or Mflops.shape[1] != self.n_targets):
            print("This Data is not compatible with the Training Data")
            return [], []
        else:
            
            ## Compute Relative Variances of experiments
            self.relVar[1] = np.array([])
            for i in range(Mflops.shape[0]):
                perfs = Mflops[i, :int(min(Xdata[i, 3], Mflops.shape[1]))]
                self.relVar[1] = np.append(self.relVar[1], (np.var(perfs) / np.mean(perfs) ) )
                
            ## Optimisation step standardization and evaluation
            self.optimize(Xdata, Mflops, index = 1)
    
            ## Output accuracies
            predPerfs, predCs, overhead = self.predict(Xdata)
            return self.score(predPerfs, predCs, Mflops, printRes), overhead
        
            
        
        
        
    
        
        
        
        