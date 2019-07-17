#!/bin/bash
blazemark_dir="/home/glaberge/opt/master/blazemark"

benchmarks=('dmattdmatmult')
threads=('4' '6' '8' '10' '12' '14' '16')
chunk_sizes=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '20' '30' '40' '50' '60' '70' '80' '90' '100' )
#'200' '300' '400' '500' '600' '700' '800' '900' '1000')
block_size_vec=('16')
block_size_row=('64')
block_size_col=('64')

touch temp_file.txt
echo "" > temp_file.txt

for c in "${chunk_sizes[@]}"
do  
    echo "### CS=${c} ###" >> temp_file.txt

    #update chun_size
    ./update_params.sh 1 ${block_size_vec[0]} $c ${block_size_row[0]} ${block_size_col[0]}

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
