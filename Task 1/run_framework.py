# record_of_all_fitnesses_each_generation stores all fitnesses at all time points

import os
import sys
from Framework.Experiment import Experiment

algorithm = 'GA'
experiment_name = '{}_{}_{}'.format(algorithm, sys.argv[1], sys.argv[2])

"""
All algorithms should use the following parameters
population_size : 100
generations: 100 (fixed) (max_fitness_evaluations)
number of runs per enemy per algorithm: 10
number of neurons: 50
enemy numbers: 5,6 & 7

Island:
islands: 4
migrations size: 2
generations per migration: 10

"""

if algorithm == 'NEAT':
    parameters = {
        'enemies': sys.argv[1],
        'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
        'generations_while_not_improving': 25,
        'max_fitness_evaluations' : 100
    }

if algorithm == 'GA':
    # The code should be executed by running: python run_framework.py [enemy number] [number of the simulation]
    parameters = {
        'enemies': sys.argv[1], # run python run_framework.py 5,6and7
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : 'replace worst',
        'max_fitness_evaluations' : 100,
        'hidden_neurons' : 50,
        'population_size' : 100,  # > tournament_size * parents_per_offspring
        'edge_domain' : [-1, 1],
        'tournament_size' : 2,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 2  # amount of children per breeding group
    }

if algorithm == 'Island':
    parameters = {
        'num_islands' : 4, # > 1, else gets stuck in a while statement in IslandExperiment.migrate
        'migrations' : 50,
        'migration_size': 2,
        'migration_type': 'copy', # exchange or copy
        'evaluations_before_migration' : 1,
        'enemies': sys.argv[1],
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : 'replace worst',
        'max_fitness_evaluations' : 100,
        'hidden_neurons' : 50,
        'population_size' : 100,  # > tournament_size * parents_per_offspring
        'edge_domain' : [-1, 1],
        'tournament_size' : 1,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 2  # amount of children per breeding group
    }

e = Experiment(experiment_name, algorithm, parameters)