#!/bin/bash

FN=$1

#turn flourescnse -> spikes
python oopsi.py $FN.txt

#turn spikes -> tensor
python tensify.py $FN.deconvolved.dat

#factorize
factorization=$(python ../bptf/code/bptf.py -d $FN.dtensor.dat -o $FN.out -v -k=25 -a=.1 -t=1e-5)

#results
python results.py $factorization
