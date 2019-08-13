#!/bin/bash
blazemark_dir="/home/glaberge/opt/master/blazemark"

bench=$1
threads=('1' '2' '4' '6' '8' '10' '12' '14' '16')

touch "${1}.txt"
cd $blazemark_dir
make $bench
cd -

#run on all threads candidates
for thr in "${threads[@]}"
do
    echo "--- ${thr} THREADS ---" >> "${1}.txt"
    
    ${blazemark_dir}/bin/${bench} -only-blaze --hpx:threads=${thr} >> "${1}.txt"
done
