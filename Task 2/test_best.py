import pickle
import os
import sys
import numpy as np
from Framework.Experiment import Experiment
"""
    Readme
    Use as follows:
    python train_against_single_enemy.py algorithm enemy trial resume

    algorithm is GA, NEAT, Island or TestStochasticity
    enemy is any integer from 1 through 8
    trial is any string
    resume is 0 or 1
"""

algorithm = sys.argv[1]
enemy = sys.argv[2]
enemy = [int(s) for s in enemy.split(',')]

best_genomes = []
for en in range(1, 8+1):
    fitness_list = []
    gains_list = []
    player_lifes_all = []
    enemy_lifes_all = []

    fitness_enemy_list = []
    gains_enemy_list = []
    player_lifes = []
    enemy_lifes_list = []
    for trial in range(1, 11):
        experiment_name = '{}_{}_{}'.format(algorithm, enemy, trial)

        best_model = pickle.load(
            open("{}/best.pkl".format(experiment_name), "rb"))
        if algorithm == 'NEAT':
            parameters = {
                'enemies': enemy,
                'enemy_amount': 8,
                'multiplemode': "yes",
                'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
                'max_fitness_evaluations': 50,
                'resume': 0
            }

        elif algorithm == 'GA_package':
            parameters = {
                'enemies': enemy,
                'enemy_amount': 8,
                'multiplemode': "yes",
                'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-GA_package'),
                'max_fitness_evaluations': 50,
                'resume': 0
            }
        elif algorithm == 'Island':
            parameters = {
                'num_islands': 3,  # > 1, else gets stuck in a while statement in IslandExperiment.migrate
                'migrations': 5,
                'migration_size': 2,
                'migration_type': 'copy',  # exchange or copy
                'enemies': 1,
                'enemy_amount': 3,
                'multiplemode': "yes",
                'parent_selection_type': 'tournament',
                'keep_best_solution': True,
                # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
                'fitness_order': [0, 'STOP'],
                # set the value which the mean must exceed before the next kind of fitness comes in use
                'fitness_threshold': [100, 100],
                'crossover_weight': 'random',
                'survival_mechanism': 'replace worst',
                'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-GA_package'),
                'max_fitness_evaluations': 50,
                'hidden_neurons': 10,
                'population_size': 51,
                'edge_domain': [-1, 1],
                'tournament_size': 2,
                'parents_per_offspring': 2,
                'mutation_probability': .2,
                'reproductivity': 2,  # amount of children per breeding group
                'resume': 0
            }
        parameters['trial'] = trial
        parameters['multiplemode'] = "no"
        parameters['enemies'] = [en]
        parameters['test_model'] = best_model

        f = []
        g = []
        player_lifes = []
        enemy_lifes = []
        for _ in range(5):
            e = Experiment(experiment_name, algorithm, parameters)
            fitness, gain, player_life, enemy_life = e.test()
            f.append(fitness)
            g.append(gain)
            player_lifes.append(player_life)
            enemy_lifes.append(enemy_life)
        fitness_enemy_list.append(np.mean(f))
        gains_enemy_list.append(np.mean(g))
        player_lifes_all.append(np.mean(player_lifes))
        enemy_lifes_all.append(np.mean(enemy_lifes))
    fitness_list.append(fitness_enemy_list)
    gains_list.append(gains_enemy_list)

    print(en, "fitness", np.mean(fitness_list), np.std(fitness_list))
    print(en, "gains", np.mean(gains_list), np.std(gains_list))
    print(en, "player life", np.mean(player_lifes_all), np.std(player_lifes_all))
    print(en, "enemy life", np.mean(enemy_lifes_all), np.std(enemy_lifes_all))
