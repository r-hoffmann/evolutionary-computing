for ((T=1;T<=10;T++)); do
    python train_against_multiple_enemies.py NEAT 1,2,3,4,5,6,7,8 "yes" 3 $T 0 | tee "NEAT_trial_$T.log"
    wait 
done
