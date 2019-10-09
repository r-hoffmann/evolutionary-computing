#!/bin/bash
for algorithm in 'GA' 'Island' 'NEAT' ; do
    for ((E=5;E<=8;E++)); do
        echo "Started $algorithm on enemy $E.";
        python train_against_single_enemy.py $algorithm $E 1;
        wait 
    done
    echo "All experiments with $algorithm are finished.";
done