# I do wanna make the child a class straight away and inherit the mutation of the parents

# import packages
import os, random, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from Framework.Algorithm import Algorithm
from Framework.Individual import Individual

class GeneticAlgorithmIC(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mating_size = self.parameters['parents_per_offspring']
        self.enemies = [self.parameters['enemies']]
        super().__init__(parameters)
        self.edges = 20 * self.parameters['hidden_neurons'] + 5 * self.parameters['hidden_neurons']
        self.population = []
        self.children = []
        self.record_of_all_fitnesses_each_generation = []

    def run(self):
        print('self.experiment_name=%s'%self.experiment_name)
        # initialise the simulation
        self.init_run()
        self.evaluation_nr = 0
        # store the initial fitnesses
        generations_fitnesses = []
        for individual in self.population:
            generations_fitnesses.append(list(individual.fitness))
        self.record_of_all_fitnesses_each_generation.append(generations_fitnesses)
        # run the simulation until a stop condition is reached
        while self.stop_condition():
            print('started evaluation number %i' % self.evaluation_nr)
            self.step()
            self.evaluation_nr += 1
            generations_fitnesses = []
            #store the fitnesses of the generation
            for individual in self.population:
                generations_fitnesses.append(list(individual.fitness))
            self.record_of_all_fitnesses_each_generation.append(generations_fitnesses)

        self.save_results()

        # plot the results to get an impression of possible improvements
        #mean_std_max = self.obtain_mean_std_max()
        #self.plot_fitness(mean_std_max)

    def init_run(self):
        # set the first fitness type to select on
        self.definition_nr = 0
        self.fitness_definition = self.parameters['fitness_order'][self.definition_nr]
        # generate an initial population
        for _ in range(self.parameters['population_size']):
            # add an individual
            self.population.append(Individual())
            # give the individual a random network
            self.population[_].network = np.random.uniform(self.parameters['edge_domain'][0], self.parameters['edge_domain'][1],self.edges)
            # determine the fitness of this network
            self.population[_] = self.obtain_fitness([self.population[_]])[0]
            # set the mutation rate
            if self.parameters['mutation_probability'] == 'flexible':
                self.population[_].mutation_rate = .2
            else:
                self.population[_].mutation_rate = self.parameters['mutation_probability']


    # don't do another generation if the stop condition is reached
    def stop_condition(self):
        return self.fitness_definition != 'STOP' and self.evaluation_nr < self.parameters['max_fitness_evaluations']

    # taking a generation step
    def step(self):
        # extract the networks of the population that won in sexual selection
        parents = self.parent_selection()
        # generate networks of children and give them a class
        unmutated_children = self.recombination(parents)
        # mutate the childrens networks
        mutated_children = self.mutation(unmutated_children)
        # change the population to only consist of children (class)
        self.population = self.obtain_fitness(mutated_children)
        # remove unfit individuals from the population
        self.survivor_selection()

    def parent_selection(self):
        # select the parents
        if self.parameters['parent_selection_type'] == 'tournament':
            reproductive_individuals = self.tournament()
        else:
            print('Error: no appropriate parent selection method selected')
        return reproductive_individuals

    # perform a tournament to choose the parents that reproduce
    def tournament(self):
        reproductive_individuals = []
        # match up individuals for tournament
        # randomize the order of the population
        random.shuffle(self.population)
        for tournament_number in range(int(self.parameters['population_size']/self.parameters['tournament_size'])):
            fitnesses_tournament = []
            for tournament_pos in range(self.parameters['tournament_size']):
                individual_nr = tournament_number*self.parameters['tournament_size']+tournament_pos
                fitnesses_tournament.append(self.population[individual_nr].fitness[self.definition_nr])
            winner_pos = fitnesses_tournament.index(max(fitnesses_tournament))
            reproductive_individuals.append(self.population[winner_pos+tournament_number])
        return reproductive_individuals

    # create the children from the selected parents
    def recombination(self,parents):
        unmutated_networks = []
        mutation_rates = []
        # go through the breeding groups / pairs
        for breeding_group in range(int(len(parents)/self.mating_size)):
            # make a list of the networks to cross over
            picked_parents = parents[breeding_group*self.mating_size:breeding_group*self.mating_size+self.mating_size]
            for _ in range(self.parameters['reproductivity']):
                # initiate child as list of zeros of the same length as the information contained in a single parent
                child_network = np.zeros(len(picked_parents[0].network))
                # go through all genes
                for gene_nr in range(len(picked_parents[0].network)):
                    if self.parameters['crossover_weight'] == 'random':
                        # make a list of heritability strengths summing to 1
                        heritabilities = []
                        devidable_proportion = 1
                        for parent_nr in range(len(picked_parents) - 1):
                            inheritance = np.random.rand() * devidable_proportion
                            # give child proportional part of parent value
                            heritabilities.append(inheritance)
                            devidable_proportion -= inheritance
                        # the last heritability is
                        heritabilities.append(devidable_proportion)
                        # randomize the heritabilities to prevent a parent from dominating the offspring values
                        random.shuffle(heritabilities)
                        # adapt the weight in the child's network
                        for parent_nr in range(len(picked_parents)):
                            child_network[gene_nr] += picked_parents[parent_nr].network[gene_nr] * heritabilities[parent_nr]
                    else: print('crossover_weight is not defined properly and should probably be \'random\'')
                unmutated_networks.append(child_network)
                # recombine the mutation rate PRESENTLY ALWAYS RANDOM WEIGHT
                heritabilities = []
                for parent_nr in range(len(picked_parents)):
                    heritabilities.append(np.random.rand())
                child_mutation_rate = 0
                for parent_nr in range(len(heritabilities)):
                    child_mutation_rate += heritabilities[parent_nr]/sum(heritabilities) * picked_parents[parent_nr].mutation_rate
                mutation_rates.append(child_mutation_rate)
        # make classes of the children
        children = []
        for _ in range(len(unmutated_networks)):
            children.append(Individual())
            children[_].network = unmutated_networks[_]
            children[_].mutation_rate = mutation_rates[_]
        return children

    # mutate the genes and mutation rate of the children
    def mutation(self,unmutated_children):
        for child in unmutated_children:
            # go through all genes of the child
            for gene_nr in range(len(child.network)):
                # mutate of random number is smaller than mutation probability
                if np.random.rand() < child.mutation_rate:
                    # only accept new values if they are in the accepted domain
                    mutated_allele = child.network[gene_nr] + np.random.normal(0, 1)
                    while not(self.parameters['edge_domain'][0] < mutated_allele < self.parameters['edge_domain'][1]):
                        mutated_allele = child.network[gene_nr] + np.random.normal(0, 1)
                    child.network[gene_nr] = mutated_allele
            # mutate the mutation rate
            if self.parameters['mutation_probability'] == 'flexible':
                if np.random.rand() < child.mutation_rate:
                    mutated_rate = -1
                    while not(0 < mutated_rate < 1):
                        mutated_rate = child.mutation_rate + np.random.normal(0, .1)
                    child.mutation_rate = mutated_rate
        return unmutated_children # which are now mutated

    # from the networks created by recombination and mutation make class individuals
    def obtain_fitness(self,mutated_children):
        for _ in range(len(mutated_children)):
            mutated_children[_].fitness = self.determine_fitness([mutated_children[_].network])[0]
        return mutated_children

    # select the survivors to form the next generation
    def survivor_selection(self):
        if self.parameters['survival_mechanism'] == '(μ, λ) Selection':
            fitnesses = []
            for individual in self.population:
                fitnesses.append(individual.fitness[self.fitness_definition])
            # order the fitness scores from highest to lowest
            fitnesses.sort(reverse=True)
            # select the 100th fittest individual
            fitness_threshold = fitnesses[self.parameters['population_size']-1]
            # remove all individuals that don't reach the fitness threshold until the population size is back to normal
            _ = 0
            while len(self.population) > self.parameters['population_size']:
                if self.population[_].fitness[self.fitness_definition] <= fitness_threshold:
                    del self.population[_]
                    #self.population.remove(self.population[_])
                else:
                    _ += 1

    # obsolete function containing discontinued functionalities
    """
    def survivor_selection(self):
        # add the children at the end of the population array
        oversized_population = np.concatenate((self.survived_population, children))

        # add the children's fitnesses at the end of the population_fitness array
        new_population_fitness = np.concatenate((self.survived_fitnesses, self.fitness_children))
        # remove the appropriate amount of individuals to sustain a fixed population size
        self.survived_fitnesses, self.survived_population = self.live_and_let_die(new_population_fitness,
                                                                                  oversized_population)

        # store the fitness- mean, standard deviation and maximum for plotting
        self.fitness_record = np.append(self.fitness_record, self.save_fitness(self.survived_fitnesses), axis=0)
        # if the mean fitness score exceeds a preselected numer, change the fitness score used
        if self.fitness_record[self.evaluation_nr + 1, 0, self.definition_nr] > self.parameters['fitness_threshold'][
            self.definition_nr]:
            self.definition_nr += 1
            self.fitness_definition = self.parameters['fitness_order'][self.definition_nr]
        # increase the evaluation number with 1
        self.evaluation_nr += 1

    # obsolete function containing discontinued functionalities
    # select the individuals to continue to the next generation
    def live_and_let_die(self, fitnesses, population):
        # reduce population to desired population size
        survival_scores = []
        if self.parameters['survival_mechanism'] == 'weighted probability':
            for individual in fitnesses:
                # give each individual a survival score based on their fitness and a  random number
                # add 1 to make sure not most of them are 0
                survival_scores.append(np.random.rand()*(individual[self.fitness_definition]+1))
        elif self.parameters['survival_mechanism'] == 'replace worst':
            for individual in fitnesses:
                survival_scores.append(individual[self.fitness_definition] + 1)
        if self.parameters['keep_best_solution']:
            # change the survival score of the fittest individual to the highest
            index_topfit = np.argmax(fitnesses[:,self.fitness_definition])
            survival_scores[index_topfit] = max(survival_scores) + 1
        # determine the fitness value of the ordered population of the individual at the population size
        ordered_survival_scores = survival_scores[:]
        ordered_survival_scores.sort(reverse=True)
        survival_threshold = ordered_survival_scores[self.parameters['population_size']]
        individual_nr = 0
        # remove individuals with a too low survival score, also removing their fitness and survival score
        while self.parameters['population_size'] < len(population):
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

    """

    def determine_unique_numbers(self, array):
        # store the amount of unique elements per column
        unique_elements = []
        for column_nr in range(len(array[0])):
            set_column = set(array[:, column_nr])
            unique_list = list(set_column)
            unique_elements.append(len(unique_list))
        return unique_elements

    def save_results(self):
        # save a record of all fitnesses of all individuals in all generations to a pickle file
        pickle_out = open(
            '' + self.experiment_name + '/fitness_record_GA_enemy' + sys.argv[1] + '_run' + sys.argv[2] + '.pickle',
            'wb')
        pickle.dump(self.record_of_all_fitnesses_each_generation, pickle_out)
        pickle_out.close()

        # print('self.record_of_all_fitnesses_each_generation=\n',self.record_of_all_fitnesses_each_generation)

        # save the best solution
        fitnesses = []
        for individual in self.population:
            fitnesses.append(individual.fitness[self.fitness_definition])
        fittest_network = self.population[fitnesses.index(max(fitnesses))].network
        pickle_out = open(
            '' + self.experiment_name + '/best_solution_GA_enemy' + sys.argv[1] + '_run' + sys.argv[2] + '.pickle',
            'wb')
        pickle.dump(fittest_network, pickle_out)
        pickle_out.close()

        # save mutation rates
        mutation_rates = []
        for individual in self.population:
            mutation_rates.append(individual.mutation_rate)
        pickle_out = open(
            '' + self.experiment_name + '/mutation_rates_GA_enemy' + sys.argv[1] + '_run' + sys.argv[2] + '.pickle',
            'wb')
        pickle.dump(mutation_rates, pickle_out)
        pickle_out.close()

    def obtain_mean_std_max(self):
        fitnesses_statistics = []
        for generation in self.record_of_all_fitnesses_each_generation:
            generation_fitnesses = []
            for individual in generation:
                # use the default fitness value
                generation_fitnesses.append(individual[0])
            mean_fitn = np.mean(generation_fitnesses)
            std_fitn = np.std(generation_fitnesses)
            max_fitn = max(generation_fitnesses)
            fitnesses_statistics.append([mean_fitn, std_fitn, max_fitn])
        return np.array(fitnesses_statistics)

    def plot_fitness(self,analysed_fitnesses):
        # create lists of mean plus and minus standard deviations
        std_mean = [[], [], []]
        for time_point in analysed_fitnesses:
            for pos_neg in [-1, 1]:
                std_mean[pos_neg].append(time_point[0] + pos_neg * time_point[1])
        plt.plot(analysed_fitnesses[:, 0], label='average')
        plt.plot(std_mean[-1], label='-1 sd')
        plt.plot(std_mean[1], label='+1 sd')
        plt.plot(analysed_fitnesses[:, 2], label='best')
        plt.legend()  # bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.) #outside the frame
        plt.xlabel('fitness evaluation')
        plt.ylabel('fitness score')
        #plt.savefig('task_2_GA_' +  sys.argv[1] + '/fitness_record_GA_enemy' + sys.argv[1]+'_run' + sys.argv[2] + '.png')
        plt.show()
        plt.close()