###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import os, sys, time
import numpy as np
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import task_1_GA_controller

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# runs simulation
def simulation(env, x):
    f,_,_,_ = env.play(pcont=x)
    return f

for enemy in range(1, 9):
    experiment_name = 'task_1_GA_{}'.format(enemy)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=task_1_GA_controller(),
                    enemymode="static",
                    level=2,
                    speed="normal")

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print('Best AI against enemy {}\n'.format(enemy))
    evaluate([bsol])