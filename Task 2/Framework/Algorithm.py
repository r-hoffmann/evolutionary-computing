import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'evoman')
sys.path.insert(0, '..')
from environment_without_rendering import Environment

class Algorithm:
    def __init__(self, parameters):
        self.parameters = parameters
        self.experiment_name = self.parameters['experiment_name']
        # initializes environment with ai player using random controller, playing against static enemy
        self.env = Environment(experiment_name=self.parameters['experiment_name'],
                  enemies=self.parameters['enemies'],
                  playermode="ai",
                  player_controller=self.parameters['player_controller'],
                  enemymode="static",
                  level=2,
                  speed="fastest")

    def play_evoman(self, x):
        fitness_scores = self.env.play(pcont=x)
        return fitness_scores

    # determines the fitness of the population
    # if fitness is negative it is corrected to 1
    # enemy fitness score = (100 - enemy life)
    def determine_fitness(self, x):
        population_fitnesses = np.array(list(map(lambda y: self.play_evoman(y), x)))
        # add column to add fitness score type
        z = np.zeros((population_fitnesses.shape[0], 1))
        population_fitnesses = np.append(population_fitnesses, z, axis=1)
        for individual_fitnesses in population_fitnesses:
            if individual_fitnesses[0] < 1:
                individual_fitnesses[0] = 1
            # rotate scores such that decreasing enemy life increases the score
            individual_fitnesses[2] = 100 - individual_fitnesses[2]
            # add aditional score .9 (100 - enemy life) + .1 player life
            individual_fitnesses[4] = .9 * individual_fitnesses[2] + .1 * individual_fitnesses[1]
        return population_fitnesses

    # plot the fitness development
    # input is a list as created by save_fitness(), but choose a fitness measure
    def plot_fitness(self):
        fitness_row = self.fitness_record[:,:,0]
        # create lists of mean plus and minus standard deviations
        std_mean = [[], [], []]
        for time_point in fitness_row:
            for pos_neg in [-1, 1]:
                std_mean[pos_neg].append(time_point[0] + pos_neg * time_point[1])
        plt.plot(fitness_row[:, 0], label='average')
        plt.plot(std_mean[-1], label='-1 sd')
        plt.plot(std_mean[1], label='+1 sd')
        plt.plot(fitness_row[:, 2], label='best')
        plt.legend()  # bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.) #outside the frame
        plt.xlabel('fitness evaluation')
        plt.ylabel('fitness score')
        plt.savefig('task_1_GA_' +  sys.argv[1] + '/fitness_record_GA_enemy' + sys.argv[1]+'_run' + sys.argv[2] + '.png')
        plt.close()
        #plt.show()