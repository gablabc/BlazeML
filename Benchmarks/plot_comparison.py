#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:35:13 2018

@author: gabriel
"""


import numpy as np
import matplotlib.pyplot as plt
import sys

## Latex Font ##
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

 
###read inputs
#benchmark = sys.argv[1]
#threads = sys.argv[2]
#path_to_figures = sys.argv[3]


benchmark = 'dmattdmatmult'
threads = "14"


legend = []
for platform in ['HPX_1.3.0' , 'MachineLearning', 'Random']:
    filename= platform + "/" + benchmark + ".dat"
    
    lines=[]
    with open(filename,"r") as file:
        for line in file:
            lines.append(line.split("\n")[0])
            
    plt.figure(1, figsize=[9, 7])        
    ###plot benchmarks
    read_index = 0
    while lines[read_index] != "--- " + str(threads) + " THREADS ---":
        read_index += 1
    #reached the right data
    read_index += 4
    x = []
    y = []

    while lines[read_index][0:3] != "---":
        numbers=lines[read_index].split(" ")
        x.append(float(numbers[0]))
        y.append(float(numbers[1]))
        read_index += 1
        if read_index>len(lines) - 1:
            break
    plt.plot(x, y)
        
plt.legend(["HPX 1.3.0", "MachineLearning", "Random"],framealpha = 1)
plt.xscale("log")    
plt.grid(True, which = "both", ls = "-", alpha = 0.5)     
plt.ylabel("Mflop/s")
if benchmark == "daxpy" or benchmark == "dvecdvecadd":
    plt.xlabel("Vector Size")
else:
    plt.xlabel("Matrix Size")
#plt.title('Performance comparison on benchmark : ' + str(benchmark) + ' on ' + str(threads) + ' CPU''s')
#plt.savefig(path_to_figures + "/" + benchmark + "/" + benchmark + "_" + threads + "CPU.pdf")
