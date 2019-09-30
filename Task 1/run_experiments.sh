#!/bin/bash
experiment='NEAT';
enemy=1;

for ((N=1;N<=5;N++)); do
	python run_framework.py 1 $N &
    echo "Started $experiment on $enemy. Trial $N.";
done