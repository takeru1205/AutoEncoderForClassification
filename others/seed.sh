#!/bin/zsh

for EP in 30 42 50
do
	echo "Train 3ensembel.py SEED $EP"
	python 3ensemble.py --seed $EP --ae-epoch 40 --train-epoch 40
done

