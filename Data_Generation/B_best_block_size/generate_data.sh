#!/bin/bash
blazemark_dir="/home/glaberge/opt/master/blazemark"

benchmarks=('dmatdvecmult')
threads=('4' '8' '12')
#'4' '6' '8' '10' '12' '14' '16')
chunk_sizes=('1') #'5' '10' '15' '20' '25')
block_size_vec=('4' '16' '64' '128' '256' '512')
block_size_row=('4' '1024' '64' '64' '1024' '256')
block_size_col=('1024' '4' '64' '1024' '64' '256')

touch temp_file.txt
echo "" > temp_file.txt

for b in `seq 0 5`
do  
    echo "### B_row=1 ###" >> temp_file.txt
    echo "### B_col=${block_size_vec[$b]} ###" >> temp_file.txt

    #update chun_size
    ./update_params.sh 1 ${block_size_vec[$b]} ${chunk_sizes[0]} ${block_size_row[$b]} ${block_size_col[$b]}

    #recompile benchmarks with new parameters
    cd ${blazemark_dir}
    for ben in "${benchmarks[@]}"
    do
        make ${ben}
    done
    cd -
  
    #run on all threads candidates
    for thr in "${threads[@]}"
    do
        echo "--- ${thr} THREADS ---" >> temp_file.txt
        
        for ben in "${benchmarks[@]}"
        do
            ${blazemark_dir}/bin/${ben} -only-blaze --hpx:threads=${thr} >> temp_file.txt
            echo "done" >> temp_file.txt
        done
    done
done

#./convertFile
#rm temp_file.txt
