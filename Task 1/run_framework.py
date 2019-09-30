# record_of_all_fitnesses_each_generation stores all fitnesses at all time points

import os
import sys
from Framework.Experiment import Experiment

algorithm = 'NEAT'
experiment_name = '{}_{}_{}'.format(algorithm, sys.argv[1], sys.argv[2])

if algorithm == 'NEAT':
    parameters = {
        'enemies': sys.argv[1],
        'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
        'generations_while_not_improving': 25
    }

if algorithm == 'GA':
    # The code should be executed by running: python run_framework.py [enemy number] [number of the simulation]
    parameters = {
        'enemies': sys.argv[1],
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [100, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : 'replace worst',
        'max_fitness_evaluations' : 50,
        'hidden_neurons' : 10,
        'population_size' : 100,  # > tournament_size * parents_per_offspring
        'edge_domain' : [-1, 1],
        'tournament_size' : 1,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 2  # amount of children per breeding group
    }



if algorithm == 'Island':
    parameters = {
        'num_islands' : 2, # > 1, else gets stuck in a while statement in IslandExperiment.migrate
        'migrations' : 5,
        'evaluations_before_migration' : 1,
        'enemies': [1],
        'parent_selection_type': 'tournament',
        'keep_best_solution' : True,
        'fitness_order' : [2, 4, 0, 'STOP'],  # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        'fitness_threshold': [60, 60, 75, 100],  # set the value which the mean must exceed before the next kind of fitness comes in use
        'crossover_weight' : 'random',
        'survival_mechanism' : 'replace worst',
        'max_fitness_evaluations' : 1,
        'hidden_neurons' : 1,
        'population_size' : 2,  # > tournament_size * parents_per_offspring
        'edge_domain' : [-1, 1],
        'tournament_size' : 1,
        'parents_per_offspring' : 2,
        'mutation_probability' : .2,
        'reproductivity' : 2  # amount of children per breeding group
    }

e = Experiment(experiment_name, algorithm, parameters)