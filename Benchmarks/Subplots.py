#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:59:28 2019

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


benchmarks = ['daxpy', 'dmatdvecmult', 'tdmattdmatadd', 'dmattdmatmult']
platforms = ['HPX_1.3.0' , 'MachineLearning', 'EqualShare']
symbols = ['--', '-', ':']
linewidths = [2.5, 1, 2.5]
xranges = [[25000, 2500000], [100, 2500], [100, 2100], [50, 10000]]
yranges = [[0, 8000], [0, 12000], [0, 12000],
           [0, 25000], [0, 30000], [0, 40000],
           [0, 3000], [0, 5000], [0, 4000],
           [0, 60000], [0, 100000], [0, 150000]]
threads = [4, 8, 12]

index = 1
fig = plt.figure(1, figsize = [20 ,12])
for i in range(len(benchmarks)):
    for thread in threads:    
        ax = fig.add_subplot(4, 3, index)
        for j in range(len(platforms)):
            filename= platforms[j] + "/" + benchmarks[i] + ".dat"
            lines=[]
            with open(filename,"r") as file:
                for line in file:
                    lines.append(line.split("\n")[0])
            
           
            ###plot benchmarks
            read_index = 0
            while lines[read_index] != "--- " + str(thread) + " THREADS ---":
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
            ax.plot(x, y, "k" + symbols[j], linewidth = linewidths[j])
            ax.set_xlim((xranges[i][0], xranges[i][1]))
            ax.set_ylim((yranges[index - 1][0], yranges[index - 1][1]))
            ax.set_xscale("log")
            ax.grid(True, which = "both", ls = "-", alpha = 0.5)   
        index += 1
fig.legend(ax.get_lines(), ['Old', 'ML', 'EqualShare'], loc = (0.73, 0.05), ncol=3)
fig.text(0.07, 0.525, "Mflop/s", ha = 'center', rotation = 'vertical', size = 15)

fig.text(0.04, 0.87, benchmarks[0], ha = 'center', rotation = 'vertical', size = 15)
fig.text(0.04, 0.59, benchmarks[1], ha = 'center', rotation = 'vertical', size = 15)
fig.text(0.04, 0.37, benchmarks[2], ha = 'center', rotation = 'vertical', size = 15)
fig.text(0.04, 0.13, benchmarks[3], ha = 'center', rotation = 'vertical', size = 15)

fig.text(0.24, 0.01, "4 threads", ha = 'center', size = 15)
fig.text(0.52, 0.01, "8 threads", ha = 'center', size = 15)
fig.text(0.52, 0.05, "matrix (vector) size", ha = 'center', size = 15)
fig.text(0.80, 0.01, "12 threads", ha = 'center', size = 15)
plt.subplots_adjust(top=0.98)
#plt.grid(True, which = "both", ls = "-", alpha = 0.5)     
#plt.title('Performance comparison on benchmark : ' + str(benchmark) + ' on ' + str(threads) + ' CPU''s')
#plt.savefig(path_to_figures + "/" + benchmark + "/" + benchmark + "_" + threads + "CPU.pdf")
