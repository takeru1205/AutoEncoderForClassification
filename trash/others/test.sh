#!/bin/zsh

for AE in 20 30 40 50
do 
	for TE in 20 30 40 50
	do
		for AELR in 0.0001 0.0003 0.0005 0.00075 0.001
		do
			for TRLR in 0.0001 0.0003 0.0005 0.00075 0.001
			do
				echo "Train $TE LR $TRLR, AutonEncoder $AE LR $AELR"
				python 3ensemble.py --train-epoch $TE --ae-epoch $AE --classify-lr $TRLR --ae-lr $AELR
			done
		done
	done
done


