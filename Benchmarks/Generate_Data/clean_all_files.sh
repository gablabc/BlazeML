# !\bin\bash

for i in `ls *.txt`
do

    benchmark=`echo ${i:0:-4}`
    ./file_cleaner.sh "${benchmark}.txt" "${benchmark}.dat"

done
