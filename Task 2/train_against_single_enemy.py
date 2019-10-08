import os, sys
from Framework.Experiment import Experiment
"""
    Readme
    Use as follows:
    python train_against_single_enemy.py algorithm enemy trial

    algorithm is either GA, NEAT, Island or TestStochasticity
    enemy is any integer from 1 through 8
    trial is any string.
"""

algorithm = sys.argv[1]
enemy = sys.argv[2]
trial = sys.argv[3]
max_fitness_evaluations = 1#10
hidden_neurons = 10
population_size = 4#100
experiment_name = '{}_{}_{}'.format(algorithm, enemy, trial)

if algorithm == 'NEAT':
    parameters = {
        'enemies': enemy,
        'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
        'max_fitness_evaluations' : max_fitness_evaluations
    }
elif algorithm == 'GA':
    # The code should be executed by running: python run_framework.py [enemy number] [number of the simulation]
    parameters = {
        'enemies': enemy, # run python run_framework.py 5,6and7
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
        'reproductivity' : 7  # amount of children per breeding group
    }
elif algorithm == 'Island':
    parameters = {
        'num_islands' : 4, # > 1, else gets stuck in a while statement in IslandExperiment.migrate
        'migrations' : 10,
        'migration_size': 2,
        'migration_type': 'copy', # exchange or copy
        'enemies': enemy,
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
    e = Experiment(experiment_name, algorithm, parameters)
    e.test()