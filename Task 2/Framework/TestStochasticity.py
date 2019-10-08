# create new folders by if not os.path.exists(self.experiment_name):

# record_of_all_fitnesses_each_generation stores all fitnesses of all individuals in
# import packages
import os, random, sys
import numpy as np
import sys
import pickle
from Framework.Algorithm import Algorithm

class TestStochasticity(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters)
        # set parameters
        # symbolic
        self.population_size = self.parameters['population_size']
        self.edge_domain = self.parameters['edge_domain']
        self.record_of_all_fitnesses_each_generation = []

    # generate a list of integers up to the population size
    def generate_integers(self):
        self.integer_list = []
        for integer in range(self.population_size):
            self.integer_list.append(integer)
    
    def stop_condition(self):
        return self.selection_fitness_score != 'STOP' and self.evaluation_nr < self.max_fitness_evaluations

    def init_run(self):
        # initialize population
        # set the amount of edges in the neural network
        edges = self.env.get_num_sensors() * self.parameters['hidden_neurons'] + 5 * self.parameters['hidden_neurons'] # not sure why this should be the right amount of edges
        # set the first fitness type to select on
        self.fitness_type = 0
        self.selection_fitness_score = self.parameters['fitness_order'][self.fitness_type]
        # generate an initial population
        self.survived_population = np.random.uniform(self.edge_domain[0], self.edge_domain[1], (self.population_size, edges))
        # determine and make an array of the fitnesses of the initial population
        self.survived_fitnesses = self.determine_fitness(self.survived_population)

    def step(self):
        self.survived_fitnesses = self.determine_fitness(self.survived_population)

    def run(self):
        self.init_run()
        for _ in range(self.parameters['repetitions']):
            self.step()
            self.record_of_all_fitnesses_each_generation.append(np.ndarray.tolist(self.survived_fitnesses))

        print('the fitnesses found are:\n',self.record_of_all_fitnesses_each_generation)

        #save a record of all fitnesses of all individuals in all generations to a pickle file
        pickle_out = open(''+ self.experiment_name + '/fitness_record_enemy'+sys.argv[1]+'_run'+sys.argv[2]+'.pickle', 'wb')
        pickle.dump(self.record_of_all_fitnesses_each_generation, pickle_out)
        pickle_out.close()
