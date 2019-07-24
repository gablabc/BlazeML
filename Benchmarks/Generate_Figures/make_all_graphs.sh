#!/bin/bash

platforms=('MachineLearning' 'HPX_1.3.0' )
benchmarks=('dvecdvecadd' 'daxpy' 'dmatdvecmult' 'dmatdmatadd' 
            'dmattdmatadd' 'tdmatdmatadd' 'tdmattdmatadd' 
            'dmatdmatmult' 'tdmattdmatmult' 'dmattdmatmult')
path="Figures"

for plat in "${platforms[@]}"
do 
    for ben in "${benchmarks[@]}"
    do
        python3 plot_benchmarks.py $ben $plat $path
    done
done
