import os, sys
from Framework.Experiment import Experiment
"""
    Readme
    Use as follows:
    python train_against_multiple_enemies.py algorithm enemy multiplemode enemy_amount trial

    algorithm is either GA, GA_package, NEAT, Island or TestStochasticity
    enemy is any integer from 1 through 8 or a list of multiple without [] e.g. 1,3,6
    multiplemode is "no" or "yes"
    enemy_amount is any integer from 1 through 8
    trial is any string.
    
    Multiple enemy training doesn't save yet, it only prints
    to train GA on enemy 1,3and6:
    python train_against_multiple_enemies.py GA_package 1,3,6 "yes" 3 1
    to train NEAT on enemy 1,3and6:
    python train_against_multiple_enemies.py NEAT 1,3,6 "yes" 3 1
    to train GA on all enemies at random:
    python train_against_multiple_enemies.py GA_package 1,2,3,4,5,6,7,8 "yes" 3 1
    to train NEAT on all enemies at random:
    python train_against_multiple_enemies.py NEAT 1,2,3,4,5,6,7,8 "yes" 3 1
"""

algorithm = sys.argv[1]
enemy = sys.argv[2] # will make it such that a random enemy from this list is selected
enemy = [int(s) for s in enemy.split(',')]
multiplemode= sys.argv[3] # "no" or "yes"
enemy_amount = int(sys.argv[4]) # should be 3
trial = sys.argv[5]
max_fitness_evaluations = 100
hidden_neurons = 10
population_size = 100 # for NEAT and GA_package use config file
experiment_name = '{}_{}_{}'.format(algorithm, enemy, trial)

if algorithm == 'NEAT':
    parameters = {
        'enemies': enemy,
        'enemy_amount': enemy_amount,
        'multiplemode': multiplemode,
        'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
        'max_fitness_evaluations' : max_fitness_evaluations
    }

elif algorithm == 'GA_package':
    parameters = {
        'enemies': enemy,
        'enemy_amount': enemy_amount,
        'multiplemode': multiplemode,
        'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-GA_package'),
        'max_fitness_evaluations': max_fitness_evaluations
    }

elif algorithm == 'GA':
    # The code should be executed by running: python run_framework.py [enemy number] [number of the simulation]
    parameters = {
        'enemies': enemy, # run python run_framework.py 5,6and7
        'enemy_amount': enemy_amount,
        'multiplemode': multiplemode,
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : '(μ, λ) Selection',
        'max_fitness_evaluations' : max_fitness_evaluations,
        'hidden_neurons' : hidden_neurons,
        'population_size' : population_size, 
        'edge_domain' : [-1, 1],
        'tournament_size' : 2,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 4  # amount of children per breeding group
    }
elif algorithm == 'Island':
    parameters = {
        'num_islands' : 4, # > 1, else gets stuck in a while statement in IslandExperiment.migrate
        'migrations' : 1,
        'migration_size': 2,
        'migration_type': 'copy', # exchange or copy
        'enemies': enemy,
        'enemy_amount': enemy_amount,
        'multiplemode': multiplemode,
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : 'replace worst',
        'max_fitness_evaluations' : max_fitness_evaluations,
        'hidden_neurons' : hidden_neurons,
        'population_size' : population_size, 
        'edge_domain' : [-1, 1],
        'tournament_size' : 2,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 2  # amount of children per breeding group
    }
elif algorithm == 'TestStochasticity':
    parameters = {
        'repetitions': 20,
        'enemies': enemy,  # run python run_framework.py 5,6and7
        'enemy_amount': enemy_amount,
        'multiplemode': multiplemode,
        'fitness_order': [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100], # set the value which the mean must exceed before the next kind of fitness comes in use
        'hidden_neurons': hidden_neurons,
        'population_size': population_size,  # > tournament_size * parents_per_offspring
        'edge_domain': [-1, 1],
    }

# Train
e = Experiment(experiment_name, algorithm, parameters)
best_model = e.run()

# Test
parameters['test_model'] = best_model
parameters['trained_on_enemy'] = enemy
for enemy in range(1, 8+1):
    parameters['enemies'] = str(enemy)
    parameters['multiplemode'] = "no"
    e = Experiment(experiment_name, algorithm, parameters)
    e.test()