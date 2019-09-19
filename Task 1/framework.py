################################
# From: dummy_demo.py #
################################
# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment_without_rendering import Environment

experiment_name = 'test_code'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
################################
################################
# From: task_1_GA.py #
################################
# lets individual play evoman against a static opponent
def play_evoman(env, x):
    fitness_scores = env.play(pcont=x)
    return fitness_scores

# determines the fitness of the population
# if fitness is negative it is corrected to 1
# enemy fitness score = (100 - enemy life)
def determine_fitness(x):
    population_fitnesses = np.array(list(map(lambda y: play_evoman(env,y), x)))
    for individual_fitnesses in population_fitnesses:
        if individual_fitnesses[0] < 1:
            individual_fitnesses[0] = 1
        # rotate scores such that decreasing enemy life increases the score
        individual_fitnesses[2] = 100 - individual_fitnesses[2]
    return population_fitnesses
################################

# import packages
import numpy as np
import random
import matplotlib.pyplot as plt

# set parameters
# symbolic
parent_selection_type = 'tournament'
tournament_type = 'weighted'
keep_best_solution = 'true'
selection_fitness_score = 2 # fitness = 0, player life = 1, enemy life = 2, runt time = 3
crossover_weight = 'random'
survival_mechanism = 'weighted probability'
change_fitness_function = 'True'
# numeric
max_fitness_evaluations = 1
hidden_layers = 10
population_size = 3
edge_domain = [-1,1]
tournament_size = 1
parents_per_offspring = 2
mutation_probability = .2
reproductivity = 2 # amount of children per breeding group


# generate a list of integers up to the population size
def generate_integers(population_size):
    integer_list = []
    for integer in range(population_size):
        integer_list.append(integer)
    return integer_list

#THIS FUNCTION IS NOT IN USE AT THE MOMENT
# determine the relative proportion of the fitness_score of the individuals
def determine_probabilities(population_fitness):
    fitness_sum = 0
    for individual in population_fitness:
        fitness_sum += individual[selection_fitness_score]
    probabilities = []
    for individual in population_fitness:
        selection_probability = float(individual[selection_fitness_score]/fitness_sum)
        probabilities.append(selection_probability)
    return probabilities

# perform a tournament to choose the parents that reproduce
def tournament(population_fitness,population):
    # match up individuals for tournament
    reproductive_individuals = []
    random.shuffle(integer_list) # randomize the integer_list to determine the tournament opponents
    for tournament_number in range(int(population_size/tournament_size)):
        fitnesses_tournament = []
        for individual_nr in range(tournament_size):
            shuffled_population_position = tournament_number + individual_nr
            fitnesses_tournament.append(population_fitness[integer_list[shuffled_population_position]][selection_fitness_score])
        #select winner of tournament
        #store population position of winner
        fittest_tournee = fitnesses_tournament.index(max(fitnesses_tournament))
        reproductive_individuals.append(population[integer_list[tournament_number+fittest_tournee]])
    return reproductive_individuals

# select the parents for the next population
def select_parents(population_fitness,population):
    if parent_selection_type == 'tournament':
        parents = tournament(population_fitness,population)
    else:
        print('Error: no appropriate parent selection method selected')
    return parents

# create the children from the selected parents
def breed(reproducing_individuals):
    children = []
    for breeding_group in range(int(len(reproducing_individuals)/parents_per_offspring)):
        parents = reproducing_individuals[breeding_group:breeding_group+parents_per_offspring]
        for birth in range(reproductivity):
            unmutated_child = crossover(parents)
            mutated_child = mutate(unmutated_child)
            children.append(mutated_child)
    return np.asarray(children)

# crossover the parents to create a child
def crossover(parents):
    # initiate child as list of zeros of the same length as the information contained in a single parent
    child = np.zeros(len(parents[0]))
    # go through all genes
    for gene_nr in range(len(parents[0])):
        if crossover_weight == 'random':
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
def mutate(child):
    # go through all genes of the child
    for gene_nr in range(len(child)):
        # mutate of random number is smaller than mutation probability
        if np.random.rand() < mutation_probability:
            # only accept new values if they are in the accepted domain
            mutated_allele = edge_domain[0] - 1
            while mutated_allele < edge_domain[0] or mutated_allele > edge_domain[1]:
                mutated_allele = child[gene_nr] + np.random.normal(0, 1)
            child[gene_nr] = mutated_allele
    return child

# select the individuals to continue to the next generation
def live_and_let_die(fitnesses,population):
    # reduce population to desired population size
    survival_scores = []
    if survival_mechanism == 'weighted probability':
        for individual in fitnesses:
            # give each individual a survival score based on their fitness and a  random number
            # add 1 to make sure not most of them are 0
            survival_scores.append(np.random.rand()*(individual[selection_fitness_score]+1))
    if keep_best_solution == 'true':
        # change the survival score of the fittest individual to the highest
        index_topfit = np.argmax(fitnesses[:,selection_fitness_score])
        survival_scores[index_topfit] = max(survival_scores) + 1
    # determine the fitness value of the ordered population of the individual at the population size
    ordered_survival_scores = survival_scores.copy()
    ordered_survival_scores.sort(reverse=True)
    survival_threshold = ordered_survival_scores[population_size]
    individual_nr = 0
    # remove individuals with a too low survival score, also removing their fitness and survival score
    while individual_nr < len(population):
        if survival_scores[individual_nr] <= survival_threshold:
            # remove the individuals and fitnesses fo those who died
            population = np.delete(population, individual_nr, 0)
            fitnesses = np.delete(fitnesses,individual_nr,0)
            del survival_scores[individual_nr]
        else:
            individual_nr += 1
    return fitnesses,population

# return the mean, std and max fitness
def save_fitness(fitnesses):
    mean_fitn = np.mean(fitnesses)
    std_fitn = np.std(fitnesses)
    max_fitn = max(fitnesses)
    return [mean_fitn,std_fitn,max_fitn]

# plot the fitness development
# input is a list as created by save_fitness()
def plot_fitness(fitness_record):
    # create lists of mean plus and minus standard deviations
    std_mean = [[], [], []]
    for time_point in fitness_record:
        for pos_neg in [-1, 1]:
            std_mean[pos_neg].append(time_point[0] + pos_neg * time_point[1])
    plt.plot(fitness_record[:, 0], label='average')
    plt.plot(std_mean[-1], label='-1 sd')
    plt.plot(std_mean[1], label='+1 sd')
    plt.plot(fitness_record[:, 2], label='best')
    plt.legend()  # bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.) #outside the frame
    plt.xlabel('fitness evaluation')
    plt.ylabel('fitness score')
    plt.show()


# initialize population
# make a list of integers to be able to randomize the order of the population without losing the connectedness of individuals and fitness
integer_list = generate_integers(population_size)
# set the amount of edges in the neural network
edges = (env.get_num_sensors() + 1) * hidden_layers + 5 * (hidden_layers + 1) # not sure why this should be the right amount of edges
# generate an initial population
survived_population = np.random.uniform(edge_domain[0], edge_domain[1], (population_size, edges))
# determine and make an array of the fitnesses of the initial population
survived_fitnesses = determine_fitness(survived_population)
# make an empty array to store fitness values
fitness_record = np.array([0,0,0])

# run through evaluations for a fixed amount of iterations
for evaluation in range(max_fitness_evaluations): # or while enemy is unbeaten
    # if a criterium is reached, change the fitness score
    if change_fitness_function == 'True':
        if fitness_record[evaluation+1,0] + fitness_record[evaluation+1,1] > 100:
            selection_fitness_score = 0
    # select the parents
    parents = select_parents(survived_fitnesses,survived_population)
    # make the children
    children = breed(parents)
    # evaluate the performance of the children
    fitness_children = determine_fitness(children)
    # add the children at the end of the population array
    oversized_population = np.concatenate((survived_population, children))
    # add the children's fitnesses at the end of the population_fitness array
    new_population_fitness = np.concatenate((survived_fitnesses, fitness_children))
    # remove the appropriate amount of individuals to sustain a fixed population size
    survived_fitnesses, survived_population = live_and_let_die(new_population_fitness, oversized_population)
    # store the fitness- mean, standard deviation and maximum for plotting
    fitness_record = np.vstack([fitness_record,save_fitness(survived_fitnesses[:,selection_fitness_score])])
    print(fitness_record)
    print(fitness_record[evaluation+1,0] + fitness_record[evaluation+1,1])


# plot the fitness over time
plot_fitness(fitness_record)