#!/bin/zsh

for EP in 10 20 30 42 50
do
	echo "Train main.py epoch $EP"
	python main.py --seed $EP
done

