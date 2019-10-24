import os
import pickle
import random
import numpy as np
from Framework.Algorithm import Algorithm
from Framework.GeneticAlgorithmIndividualClass import GeneticAlgorithmIC

# The corresponding algorithm.run() method must be changed to return population


class IslandAlgorithm(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        self.experiment_name = self.parameters['experiment_name']
        self.num_islands = self.parameters['num_islands']
        self.migrations = self.parameters['migrations']
        self.max_fitness_evaluations = self.parameters['max_fitness_evaluations']
        self.islands = []

    def run(self):
        for _ in range(self.num_islands):
            print('Init island {0} against enemies {1}'.format(_, self.parameters['enemies']))
            island_parameters = self.parameters.copy()
            island_parameters['enemies'] = self.parameters['enemies']
            island_parameters['population_size'] //= self.num_islands
            island = GeneticAlgorithmIC(island_parameters)
            island.init_run()
            self.islands.append(island)

        all_fitnesses = []
        for _ in range(self.migrations):
            for _ in range(self.max_fitness_evaluations // self.migrations):
                for island in self.islands:
                    island.step()
                    all_fitnesses.append(
                        [fitness for island in self.islands for fitness in island.survived_fitnesses])
            self.migrate()

        # Save to fs.
        with open('{}/all_fitnesses.pkl'.format(self.experiment_name), 'wb') as fid:
            pickle.dump(all_fitnesses, fid)

        result = sorted([[x[0], index, island] for island, island_class in enumerate(
            self.islands) for index, x in enumerate(island_class.survived_fitnesses)], reverse=True)[0]
        _, index, island = result
        fittest_individual = self.islands[island].survived_population[index]
        with open('{}/best.pkl'.format(self.experiment_name), 'wb') as fid:
            pickle.dump(fittest_individual, fid)

        return fittest_individual

    def migrate(self):
        print('migrating')
        if self.parameters['migration_type'] == 'copy':
            valid = False
            migration_island_indices = list(range(self.num_islands))
            while not valid:
                random.shuffle(migration_island_indices)
                for x, y in zip(range(self.num_islands), migration_island_indices):
                    if x == y:
                        valid = False
                        break
                valid = True

            for i, x in enumerate(migration_island_indices):
                from_island = self.islands[i]
                to_island = self.islands[x]
                migrants_indices = sorted([[x[0], i] for i, x in enumerate(
                    from_island.survived_fitnesses)], reverse=True)[:self.parameters['migration_size']]
                to_replace_indices = sorted([[x[0], i] for i, x in enumerate(
                    to_island.survived_fitnesses)])[:self.parameters['migration_size']]

                for z, y in zip(migrants_indices, to_replace_indices):
                    to_island.survived_fitnesses[y[1]
                                                 ] = to_island.survived_fitnesses[z[1]].copy()
                    to_island.survived_population[y[1]] = to_island.survived_population[z[1]].copy(
                    )
                    print('index {} of island {} replaces index {} of island {}. Fitness {}.'.format(
                        z[1], i, y[1], x, z[0]))

        elif self.parameters['migration_type'] == 'exchange':
            raise NotImplementedError()

    def test(self):
        island_parameters = self.parameters.copy()
        island_parameters['population_size'] //= self.num_islands
        example_island = GeneticAlgorithmIC(island_parameters)
        fitness, gain, player_life, enemy_life = example_island.determine_fitness_and_gain(
            self.parameters['test_model'])
        print('Test results trial {}: Fitness {}, Gain: {}.'.format(
            self.parameters['trial'], fitness, gain))
        return fitness, gain, player_life, enemy_life
