################################
# From: dummy_demo.py #
################################
# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment

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

# set parameters
# symbolic
parent_selection_type = 'tournament'
tournament_type = 'weighted'
keep_best_solution = 'true' # not yet implemented
selection_fitness_score = 2 # fitness = 0, player life = 1, enemy life = 2, runt time = 3
crossover_weight = 'random'
survival_mechanism = 'weighted probability'
# numeric
fitness_evaluations = 2
hidden_layers = 10
population_size = 8
edge_domain = [-1,1]
tournament_size = 2
parents_per_offspring = 2
mutation_probability = .2


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
    # determine the fitness value at the population size
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



# initialize population
# make a list of integers to be able to randomize the order of the population without losing the connectedness of individuals and fitness
integer_list = generate_integers(population_size)
# set the amount of edges in the neural network
edges = (env.get_num_sensors() + 1) * hidden_layers + 5 * (hidden_layers + 1) # not sure why this should be the right amount of edges
# generate an initial population
survived_population = np.random.uniform(edge_domain[0], edge_domain[1], (population_size, edges))
# determine and make an array of the fitnesses of the initial population
survived_fitnesses = determine_fitness(survived_population)

# run through evaluations for a fixed amount of iterations
for evaluation in range(fitness_evaluations): # or while enemy is unbeaten
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