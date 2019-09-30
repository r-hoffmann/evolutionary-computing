import os
import numpy as np
import random
from Framework.GeneticAlgorithm import GeneticAlgorithm
#from Framework.NeatAlgorithm import NeatAlgorithm
from Framework.PlayerNeatController import PlayerNeatController

#The corresponding algorithm.run() method must be changed to return population
class IslandExperiment:
    def __init__(self, experiment_name, parameters=dict()):
        self.experiment_name = experiment_name
        self.parameters = parameters
        self.parameters['experiment_name'] = experiment_name
        self.islands = []
        
        # if not os.path.exists(self.experiment_name):
        #     os.makedirs(self.experiment_name)

        # if algorithm=='GA':
        #     self.parameters['player_controller'] = None
        #     for i in range(num_pops):
        #         self.algorithmIslandGeneticAlgorithm(parameters, initial_pop="None")
        # elif algorithm=='NEAT':
        #     self.parameters['player_controller'] = PlayerNeatController()
        #     for i in range(num_pops):   
        #         self.algorithm{}.format(i) = NeatAlgorithm(parameters, initial_pop="none")
        # else:
        #     raise ValueError('{} is not an algorithm'.format(algorithm))
        
    # def step(self):
    #     #Run algorithm once for each population from random initialisation
    #     pop_tuple = ()
    #     for i in range(num_pops):
    #             pop{}.format() = self.algorithm{}.format(i).run()
    #             pop_tuple = pop_tuple + (pop{}.format(i),)
        
        
    #     #After a given number of generations, the population groups are completely shuffled
    #     if island_type == "Shuffle":

    #         #Each shuffled population will then initialise a new algorithm run
    #         for m in range(migrations):

    #             if m==0:
    #                 #Combine, shuffle, split
    #                 all_pops = np.concatenate(pop_tuple)
    #                 shuffled = np.random.shuffle(all_pops)
    #                 pop_list = np.split(shuffled, num_pops)
                
    #             else:
    #                 #Initialise tuple for shuffle of next meta-generation
    #                 pop_tup = ()
                
    #             #Run algorithm on each shuffled population
    #             for i in range(num_pops):
    #                 init_pop[i] = pop_list[i]
    #                 self.algorithm{}.format(i) = IslandGeneticAlgorithm(parameters, initial_pop=init_pop[i])
    #                 pop{}.format(i) = self.algorithm{}.format(i)
    #                 pop_tup = pop_tup + (pop{}.format(i),)

    #             all_pops = np.concatenate(pop_tuple)
    #             shuffled = np.random.shuffle(all_pops)
    #             pop_list = np.split(shuffled, num_pops)

    #     elif island_type == "Shuffle":

    #         #Each shuffled population will then initialise a new algorithm run
    #         for m in range(migrations):

    #             if m==0:
    #                 #Combine, shuffle, split
    #                 all_pops = np.concatenate(pop_tuple)
    #                 shuffled = np.random.shuffle(all_pops)
    #                 pop_list = np.split(shuffled, num_pops)
                
    #             else:
    #                 #Initialise tuple for shuffle of next meta-generation
    #                 pop_tup = ()
                
    #             #Run algorithm on each shuffled population
    #             for i in range(num_pops):
    #                 init_pop[i] = pop_list[i]
    #                 self.algorithm{}.format(i) = IslandGeneticAlgorithm(parameters, initial_pop=init_pop[i])
    #                 pop{}.format(i) = self.algorithm{}.format(i)
    #                 pop_tup = pop_tup + (pop{}.format(i),)

    #             all_pops = np.concatenate(pop_tuple)
    #             shuffled = np.random.shuffle(all_pops)
    #             pop_list = np.split(shuffled, num_pops)

    def run(self):

        for i in range(self.parameters['num_islands']):
            island = GeneticAlgorithm(self.parameters)
            island.init_run()
            self.islands.append(island)
        
        for m in range(self.parameters['migrations']):
            for i in range(self.parameters['evaluations_before_migration']):
                for island in self.islands:
                    island.step()
            print('Migrating...')
            self.migrate()

    def migrate(self):

        island1 = random.choice(self.islands)
        island2 = random.choice(self.islands)

        while island2 == island1:
            island2 = random.choice(self.islands)

        rand = random.sample(range(0, island1.population_size), 2)
        for i in rand:
            island2.survived_population[i], island1.survived_population[i] = island1.survived_population[i], island2.survived_population[i]
            island2.survived_fitnesses[i], island1.survived_fitnesses[i] = island1.survived_fitnesses[i], island2.survived_fitnesses[i]





            
        