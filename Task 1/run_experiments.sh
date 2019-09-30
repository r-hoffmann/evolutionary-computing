#!/bin/bash

# README: 
# This file runs 10 experiments simultaneously on default.
# This can be changed by changing MAX_N on the following line.
MAX_N=10;
# As we need to run 10 experiments per enemy, the START_N can 
# be modified to change the starting number of the simulation.
START_N=1;

read -p "Have you read the comment in this file? <y/N> " prompt
if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]
then
    experiment='NEAT';
    for ((E=1;E<=8;E++)); do
        for ((N=START_N;N<=MAX_N;N++)); do
            python run_framework.py $E $N &
            echo "Started $experiment on $E. Trial $N.";
        done
        wait 
        echo "All experiments on $enemy are finished.";
    done
else
  exit 0
fi
