import os
import numpy as np
import random
from Framework.Algorithm import Algorithm
from Framework.GeneticAlgorithm import GeneticAlgorithm

#The corresponding algorithm.run() method must be changed to return population
class IslandAlgorithm(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        self.experiment_name = self.parameters['experiment_name']
        self.num_islands = self.parameters['num_islands']
        self.migrations = self.parameters['migrations']
        self.max_fitness_evaluations = self.parameters['max_fitness_evaluations']
        self.islands = []
        
    def run(self):
        for i in range(self.num_islands):
            island = GeneticAlgorithm(self.parameters)
            island.init_run()
            self.islands.append(island)
        print('test')
        for m in range(self.migrations):
            self.migrate()
            for i in range(self.max_fitness_evaluations / self.migrations):
                print(m, i)
                for island in self.islands:
                    island.step()

    def migrate(self):
        print('migrating')
        if self.parameters['migration_type'] == 'exchange':
            valid = False
            migration_island_indices = list(range(self.num_islands))
            while not valid:
                random.shuffle(migration_island_indices)
                for x,y in zip(range(self.num_islands), migration_island_indices):
                    if x==y:
                        valid = False
                        break
                valid = True
            migration_islands = []
            for i, x in enumerate(migration_island_indices):
                from_island = self.islands[i]
                to_island = self.islands[x]
                migrants = [[x, i] for x in enumerate(from_island.survived_fitnesses)]
                print(sorted(migrants))
            

            print(migration_islands)

            for island in self.islands:
                migration_island = random.choice()
            island1 = random.choice(self.islands)
            island2 = random.choice(self.islands)
            island3 = random.choice(self.islands)
            island4 = random.choice(self.islands)

            while island2 == island1:
                island2 = random.choice(self.islands)
            
            while island3 == island1 or island3 == island2:
                island3 = random.choice(self.islands)

            while island4 == island1 or island4 == island2 or island4 == island3:
                island2 = random.choice(self.islands)

            rand = random.sample(range(0, island1.population_size), 2)
            for i in rand:
                island2.survived_population[i], island1.survived_population[i] = island1.survived_population[i], island2.survived_population[i]
                island2.survived_fitnesses[i], island1.survived_fitnesses[i] = island1.survived_fitnesses[i], island2.survived_fitnesses[i]
                island4.survived_population[i], island3.survived_population[i] = island3.survived_population[i], island4.survived_population[i]
                island4.survived_fitnesses[i], island3.survived_fitnesses[i] = island3.survived_fitnesses[i], island4.survived_fitnesses[i]
        elif self.parameters['migration_type'] == 'copy':
            pass