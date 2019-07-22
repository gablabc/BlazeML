#!/bin/env bash

#SBATCH -o glaberge_%j.out
#SBATCH -t 24:00:00
#SBATCH -p marvin
#SBATCH -N 1

./generate_data.sh $1
