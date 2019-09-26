# import packages 
import os, random, sys
import numpy as np
from Framework.Algorithm import Algorithm

class GeneticAlgorithm(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters)
        # set parameters
        # symbolic
        # TO CHANGE THE TYPE OF ENEMY: Change 'enemies' in environment_without_rendering
        self.parent_selection_type = 'tournament'
        self.keep_best_solution = True
        self.fitness_order = [2,4,0,'STOP'] # fitness = 0, player life = 1, enemy life = 2, run time = 3, lives = 4
        self.crossover_weight = 'random'
        self.survival_mechanism = 'replace worst'
        # numeric
        self.max_fitness_evaluations = 2
        self.hidden_layers = 1
        self.population_size = 4 # > tournament_size * parents_per_offspring
        self.edge_domain = [-1,1]
        self.tournament_size = 2
        self.parents_per_offspring = 2
        self.mutation_probability = .2
        self.reproductivity = 2 # amount of children per breeding group

    # generate a list of integers up to the population size
    def generate_integers(self):
        self.integer_list = []
        for integer in range(self.population_size):
            self.integer_list.append(integer)
    
    def stop_condition(self):
        return self.selection_fitness_score != 'STOP' and self.evaluation_nr < self.max_fitness_evaluations

    def run(self):
        # initialize population
        # make a list of integers to be able to randomize the order of the population without losing the connectedness of individuals and fitness
        self.generate_integers()
        # set the amount of edges in the neural network
        edges = (self.env.get_num_sensors() + 1) * self.hidden_layers + 5 * (self.hidden_layers + 1) # not sure why this should be the right amount of edges
        # set the first fitness type to select on
        self.fitness_type = 0
        self.selection_fitness_score = self.fitness_order[self.fitness_type]
        # generate an initial population
        self.survived_population = np.random.uniform(self.edge_domain[0], self.edge_domain[1], (self.population_size, edges))
        # determine and make an array of the fitnesses of the initial population
        self.survived_fitnesses = self.determine_fitness(self.survived_population)
        # make an empty array to store fitness values
        #fitness_record = np.array([0,0,0,0,0])
        # save the initial fitness
        self.fitness_record = self.save_fitness(self.survived_fitnesses)


        self.evaluation_nr = 0
        while self.stop_condition():
            self.step()
        self.plot_fitness()

    # perform a tournament to choose the parents that reproduce
    def tournament(self, population_fitness, population):
        # match up individuals for tournament
        reproductive_individuals = []
        random.shuffle(self.integer_list) # randomize the integer_list to determine the tournament opponents
        for tournament_number in range(int(self.population_size/self.tournament_size)):
            fitnesses_tournament = []
            for individual_nr in range(self.tournament_size):
                shuffled_population_position = tournament_number*self.tournament_size + individual_nr
                fitnesses_tournament.append(population_fitness[self.integer_list[shuffled_population_position]][self.selection_fitness_score])
            #select winner of tournament
            #store population position of winner
            fittest_tournee = fitnesses_tournament.index(max(fitnesses_tournament))
            reproductive_individuals.append(population[self.integer_list[tournament_number+fittest_tournee]])
        return reproductive_individuals

    # select the parents for the next population
    def select_parents(self, population_fitness, population):
        if self.parent_selection_type == 'tournament':
            parents = self.tournament(population_fitness, population)
        else:
            print('Error: no appropriate parent selection method selected')
        return parents

    # create the children from the selected parents
    def breed(self, parents):
        children = []
        for breeding_group in range(int(len(parents)/self.parents_per_offspring)):
            picked_parents = parents[breeding_group*self.parents_per_offspring:breeding_group*self.parents_per_offspring+self.parents_per_offspring]
            for _ in range(self.reproductivity):
                unmutated_child = self.crossover(picked_parents)
                mutated_child = self.mutate(unmutated_child)
                children.append(mutated_child)
        return np.asarray(children)

    # crossover the parents to create a child
    def crossover(self, parents):
        # initiate child as list of zeros of the same length as the information contained in a single parent
        child = np.zeros(len(parents[0]))
        # go through all genes
        for gene_nr in range(len(parents[0])):
            if self.crossover_weight == 'random':
                # make a list of heritability strengths summing to 1
                heritabilities = []
                devidable_proportion = 1
                for parent_nr in range(len(parents)-1):
                    inheritance = np.random.rand()*devidable_proportion
                    # give child proportional part of parent value
                    heritabilities.append(inheritance)
                    devidable_proportion -= inheritance
                heritabilities.append(devidable_proportion)
                random.shuffle(heritabilities)  # randomize the heritabilities to prevent a parent from dominating the offsrping values
                for parent_nr in range(len(parents)):
                    child[gene_nr] += parents[parent_nr][gene_nr]*heritabilities[parent_nr]
        return child

    # mutate the genes of the child
    def mutate(self, child):
        # go through all genes of the child
        for gene_nr in range(len(child)):
            # mutate of random number is smaller than mutation probability
            if np.random.rand() < self.mutation_probability:
                # only accept new values if they are in the accepted domain
                mutated_allele = self.edge_domain[0] - 1
                while not(self.edge_domain[0] < mutated_allele < self.edge_domain[1]):
                    mutated_allele = child[gene_nr] + np.random.normal(0, 1)
                child[gene_nr] = mutated_allele
        return child

    # select the individuals to continue to the next generation
    def live_and_let_die(self, fitnesses, population):
        # reduce population to desired population size
        survival_scores = []
        if self.survival_mechanism == 'weighted probability':
            for individual in fitnesses:
                # give each individual a survival score based on their fitness and a  random number
                # add 1 to make sure not most of them are 0
                survival_scores.append(np.random.rand()*(individual[self.selection_fitness_score]+1))
        elif self.survival_mechanism == 'replace worst':
            for individual in fitnesses:
                survival_scores.append(individual[self.selection_fitness_score] + 1)
        if self.keep_best_solution:
            # change the survival score of the fittest individual to the highest
            index_topfit = np.argmax(fitnesses[:,self.selection_fitness_score])
            survival_scores[index_topfit] = max(survival_scores) + 1
        # determine the fitness value of the ordered population of the individual at the population size
        ordered_survival_scores = survival_scores[:]
        ordered_survival_scores.sort(reverse=True)
        survival_threshold = ordered_survival_scores[self.population_size]
        individual_nr = 0
        # remove individuals with a too low survival score, also removing their fitness and survival score
        while self.population_size < len(population):
            if survival_scores[individual_nr] <= survival_threshold:
                # remove the individuals and fitnesses fo those who died
                population = np.delete(population, individual_nr, 0)
                fitnesses = np.delete(fitnesses,individual_nr,0)
                del survival_scores[individual_nr]
            else:
                individual_nr += 1
        return fitnesses, population

    # return the mean, std and max fitness
    def save_fitness(self, fitnesses):
        # store in colums the mean, std and max of all the 5 fitness measures in rows
        fitnesses_statistics = []
        for fitness_definition in range(fitnesses.shape[1]):
            mean_fitn = np.mean(fitnesses[:,fitness_definition])
            std_fitn = np.std(fitnesses[:,fitness_definition])
            max_fitn = max(fitnesses[:,fitness_definition])
            fitnesses_statistics.append([mean_fitn, std_fitn, max_fitn])
        # add a third dimension to be able to add new time points
        fitnesses_statistics = np.array(fitnesses_statistics)
        fitnesses_statistics = np.transpose(fitnesses_statistics)
        fitnesses_statistics = list(fitnesses_statistics)
        fitnesses_statistics = [fitnesses_statistics]
        fitnesses_statistics = np.array(fitnesses_statistics)
        return fitnesses_statistics

    def parent_selection(self):
        # select the parents
        return self.select_parents(self.survived_fitnesses, self.survived_population)

    def recombination(self, parents):
        # make the children
        children = self.breed(parents)
        # evaluate the performance of the children
        self.fitness_children = self.determine_fitness(children)
        return children

    def mutation(self, children):
        return children

    def survivor_selection(self, children):
        # add the children at the end of the population array
        oversized_population = np.concatenate((self.survived_population, children))

        # add the children's fitnesses at the end of the population_fitness array
        new_population_fitness = np.concatenate((self.survived_fitnesses, self.fitness_children))
        # remove the appropriate amount of individuals to sustain a fixed population size
        self.survived_fitnesses, self.survived_population = self.live_and_let_die(new_population_fitness, oversized_population)
        
        # store the fitness- mean, standard deviation and maximum for plotting
        self.fitness_record = np.append(self.fitness_record, self.save_fitness(self.survived_fitnesses),axis=0)
        # if a criterium is reached, change the fitness score
        if self.fitness_record[self.evaluation_nr+1,0,self.fitness_type] + self.fitness_record[self.evaluation_nr+1,1,self.fitness_type] > 100:
            self.fitness_type += 1
            self.selection_fitness_score = self.fitness_order[self.fitness_type]
            print('the fitness score now in use is %i' % self.selection_fitness_score)
        # increase the evaluation number with 1
        self.evaluation_nr += 1
        print('we are at evaluation number %i' % self.evaluation_nr)

    def determine_unique_numbers(self, array):
        # store the amount of unique elements per column
        unique_elements = []
        for column_nr in range(len(array[0])):
            set_column = set(array[:, column_nr])
            unique_list = list(set_column)
            unique_elements.append(len(unique_list))
        return unique_elements
