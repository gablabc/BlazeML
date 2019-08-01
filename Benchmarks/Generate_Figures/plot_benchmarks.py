#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:17:15 2018

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
#platform = sys.argv[2]
#path_to_figures = sys.argv[3]

### manual inputs (in script)
benchmark = 'tdmattdmatmult'
#platform = "HPX_1.3.0"
platform = 'MachineLearning'



############## importing benchmark ####################
filename=platform + "/" + benchmark + ".dat"

threads_count = [1, 2, 4, 8, 12, 16]

lines=[]
with open(filename,"r") as file:
    for line in file:
        lines.append(line.split("\n")[0])
        
plt.figure(1, figsize=[9, 7])    
    
###plot benchmarks
legend = []
read_index = 0

for threads in threads_count:
    while lines[read_index]!="--- " + str(threads) + " THREADS ---":
        read_index += 1
    #reached the right data
    read_index += 4
    benx = []
    beny = []
    
    while lines[read_index][0:3] != "---":
        numbers=lines[read_index].split(" ")
        benx.append(float(numbers[0]))
        beny.append(float(numbers[1]))
        read_index += 1
        if read_index>len(lines) - 1:
            break
    legend.append("NCPUs = " + str(threads))    
    #plot benchmarks
    plt.plot(benx, beny)



plt.xscale("log")    
plt.grid(True, which="both", ls = "-", alpha = 0.5)     
plt.ylabel("Mflop/s")
plt.legend(legend)
if benchmark == "daxpy" or benchmark == "dvecdvecadd":
    plt.xlabel("Vector Size")
else:
    plt.xlabel("Matrix Size")
    plt.legend(legend)
plt.savefig(path_to_figures + "/" + platform + "/" + benchmark + ".pdf")