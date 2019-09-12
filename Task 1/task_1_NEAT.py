"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import neat, os, sys, time, visualize
import numpy as np
sys.path.insert(0, 'evoman')
from environment import Environment
from controllers import task_1_NEAT_controller

enemy = 1

experiment_name = 'task_1_NEAT_{}'.format(enemy)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=task_1_NEAT_controller(),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params
run_mode = 'train' # train or test

# runs simulation
def get_fitness(env, controller):
    f,_,_,_ = env.play(pcont=controller)
    return f

def eval_genomes(genomes, config):
    for _, genome in genomes:
        controller = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness(env, controller)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x generations.
    winner = p.run(eval_genomes, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {
        0:'s',
        1:'s',
        2:'s',
        3:'s',
        4:'s',
        5:'s',
        6:'s',
        7:'s',
        8:'s',
        9:'s',
        10:'s',
        11:'s',
        12:'s',
        13:'s',
        14:'s',
        15:'s',
        16:'s',
        17:'s',
        18:'s',
        19:'s',
        20: 'left', 
        1: 'right', 
        2: 'jump', 
        3: 'shoot', 
        4: 'release'} 
    visualize.draw_net(config, winner, True, node_names=node_names, filename="NEAT/DiGraph")
    visualize.plot_stats(stats, ylog=False, view=True, filename='NEAT/avg_fitness.svg')
    visualize.plot_species(stats, view=True, filename='NEAT/speciation.svg')

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 100)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'NEAT', 'config-NEAT')
    run(config_path)