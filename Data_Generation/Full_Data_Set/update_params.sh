#!/bin/bash

blaze_dir="/home/glaberge/opt/master/blaze"

#the 3 params row_size column_size chunk_size 
if [ $# -eq 5 ]
then
    
    filename="${blaze_dir}/config/HPX.h"
    
    #replace vector chunk size
    cs_vec=$1
    sed -i "41s/.*/#define BLAZE_HPX_VECTOR_CHUNK_SIZE ${cs_vec}/" $filename
    echo "vector chunk size changed to ${cs_vec} "

    #replace vector block size
    b_vec=$2
    sed -i "45s/.*/#define BLAZE_HPX_VECTOR_BLOCK_SIZE ${b_vec}/" $filename
    echo "column size changed to ${b_vec} "

    #replace matrix chunk size
    cs_mat=$3
    sed -i "49s/.*/#define BLAZE_HPX_MATRIX_CHUNK_SIZE ${cs_mat}/" $filename
    echo "chunk size changed to " ${cs_mat}
 
    #replace row size
    row=$4
    sed -i "53s/.*/#define BLAZE_HPX_MATRIX_BLOCK_SIZE_ROW ${row}/" $filename
    echo "row size changed to ${row} "

    #replace column size
    column=$5
    sed -i "57s/.*/#define BLAZE_HPX_MATRIX_BLOCK_SIZE_COLUMN ${column}/" $filename
    echo "column size changed to ${column} "

else
    echo "Not right number of input parameters"
fi
