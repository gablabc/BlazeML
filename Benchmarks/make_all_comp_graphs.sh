#!/bin/bash

threads=('1' '2' '4' '6' '8' '10' '12' '14' '16')
benchmarks=('dvecdvecadd' 'daxpy' 'dmatdvecmult' 'dmatdmatadd' 
            'tdmattdmatadd' 'dmattdmatadd' 'tdmatdmatadd'
            'dmatdmatmult' 'tdmattdmatmult' 'dmattdmatmult')
path="Figures/Comparative"

for ben in "${benchmarks[@]}"
do
    mkdir "${path}/${ben}"
    for thr in "${threads[@]}"
    do
        python3 plot_comparison.py $ben $thr $path
    done
done
