#!/bin/bash

python train_against_multiple_enemies.py GA_package 1,3,6 "yes" 3 1;
echo "----------------";
wait
python train_against_multiple_enemies.py NEAT 1,3,6 "yes" 3 1;
echo "----------------";
wait
python train_against_multiple_enemies.py GA_package 1,2,3,4,5,6,7,8 "yes" 3 1;
echo "----------------";
wait
python train_against_multiple_enemies.py NEAT 1,2,3,4,5,6,7,8 "yes" 3 1;