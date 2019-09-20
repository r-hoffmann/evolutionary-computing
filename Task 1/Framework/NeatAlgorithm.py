import neat, os, visualize
from Framework.Algorithm import Algorithm

class NeatAlgorithm(Algorithm):
    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters)
        self.config_file = self.parameters['config_file']

    # runs simulation
    def determine_fitness(self, controller):
        f,_,_,_ = self.env.play(pcont=controller)
        return f

    def eval_genomes(self, genomes, config):
        for _, genome in genomes:
            controller = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.determine_fitness(controller)

    def run(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'NEAT', 'config-NEAT')

        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            self.config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoint-{}-'.format(self.parameters['enemies'][0])))

        # Run for up to x generations.
        winner = p.run(self.eval_genomes, self.parameters['generations'])

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        node_names = {
            -1:'sensor 1',
            -2:'sensor 2',
            -3:'sensor 3',
            -4:'sensor 4',
            -5:'sensor 5',
            -6:'sensor 6',
            -7:'sensor 7',
            -8:'sensor 8',
            -9:'sensor 9',
            -10:'sensor 10',
            -11:'sensor 11',
            -12:'sensor 12',
            -13:'sensor 13',
            -14:'sensor 14',
            -15:'sensor 15',
            -16:'sensor 16',
            -17:'sensor 17',
            -18:'sensor 18',
            -19:'sensor 19',
            -20:'sensor 20',
            0: 'left',
            1: 'right', 
            2: 'jump', 
            3: 'shoot', 
            4: 'release'
        }

        visualize.draw_net(config, winner, True, node_names=node_names, filename="{}/DiGraph".format(self.experiment_name))
        visualize.plot_stats(stats, ylog=False, view=True, filename='{}/avg_fitness.svg'.format(self.experiment_name))
        visualize.plot_species(stats, view=True, filename='{}/speciation.svg'.format(self.experiment_name))

        # Save to fs.
        with open('{}/best.pkl'.format(experiment_name), 'wb') as fid:
            pickle.dump(winner, fid)