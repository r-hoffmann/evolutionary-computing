#!/bin/bash
for algorithm in 'GA_package' 'Island' 'NEAT' ; do
    for ((E=5;E<=8;E++)); do
        for ((T=1;T<=10;T++)); do
            echo "Started $algorithm on enemy $E.";
            echo "python train_against_multiple_enemies.py $algorithm $E yes enemy_amount $T";
            wait 
        done
    done
    echo "All experiments with $algorithm are finished.";
done