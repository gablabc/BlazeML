# !\bin\bash

lines=`wc -l $1 | awk '{ print $1 }'`
target=$2
for i in `seq 1 $lines`
do 
    line=`sed -n "${i}p" $1`
    if [ -z "$line" ]
    then
        echo $line >> ${target}
    
    elif [ "${line:0:13}" = " Blaze kernel" ]
    then
        echo "Line removed"
    else
        echo $line >> ${target}
    fi

done
