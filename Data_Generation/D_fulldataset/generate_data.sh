#!/bin/bash
blazemark_dir="/home/glaberge/opt/master/blazemark"

benchmarks=('dmatdmatmult')
#benchmarks=('dvecdvecadd' 'dmatdvecmult' 'dmatdmatadd' 'dmatscalarmult' 'dmattdmatadd' 'dmatdmatmult')

bs_vec_tab=('1')
bs_row_tab=('256')
bs_col_tab=('256')

threads=('2' '4' '6' '8' '10' '12' '14' '16')
chunk_sizes=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '20')

touch temp_file.txt
echo "" > temp_file.txt

for cs in "${chunk_sizes[@]}"
do  
    echo "### CS=${cs} ###" >> temp_file.txt
  
    #run on all threads candidates
    for thr in "${threads[@]}"
    do
        echo "--- ${thr} THREADS ---" >> temp_file.txt
        
        #run on all benchmarks
        for i in '0'
        do

            ben=${benchmarks[$i]}
            bs_vec=${bs_vec_tab[$i]}
            bs_row=${bs_row_tab[$i]}
            bs_col=${bs_col_tab[$i]}
            
            ./update_params.sh $cs $bs_vec $cs $bs_row $bs_col

            cd "/home/glaberge/opt/master/blazemark"
            pwd
            make ${ben}
            
            cd -
            ${blazemark_dir}/bin/${ben} -only-blaze --hpx:threads=${thr} >> temp_file.txt
            echo "done" >> temp_file.txt
        done
    done
done

#./convertFile
#rm temp_file.txt
