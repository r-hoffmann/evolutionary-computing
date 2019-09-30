from __future__ import print_function
import neat, os, visualize

from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

from Framework.Algorithm import Algorithm

class SmartPopulation(neat.Population):
	def run(self, fitness_function, generations_while_not_improving=25):
		"""
		Run population until there is no better genome in last g generations
		See super class for details.
		"""

		if self.config.no_fitness_termination and (n is None):
			raise RuntimeError("Cannot have no generational limit with no fitness termination")

		k = 0
		best_genome_generations = 0
		while best_genome_generations < generations_while_not_improving:
			k += 1

			self.reporters.start_generation(self.generation)

			# Evaluate all genomes using the user-provided function.
			fitness_function(list(iteritems(self.population)), self.config)

			# Gather and report statistics.
			best = None
			for g in itervalues(self.population):
				if best is None or g.fitness > best.fitness:
					best = g
					best_genome_generations = 0
			self.reporters.post_evaluate(self.config, self.population, self.species, best)

			# Track the best genome ever seen.
			if self.best_genome is None or best.fitness > self.best_genome.fitness:
				self.best_genome = best

			if not self.config.no_fitness_termination:
				# End if the fitness threshold is reached.
				fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
				if fv >= self.config.fitness_threshold:
					self.reporters.found_solution(self.config, self.generation, best)
					break

			# Create the next generation from the current generation.
			self.population = self.reproduction.reproduce(self.config, self.species,
														  self.config.pop_size, self.generation)

			# Check for complete extinction.
			if not self.species.species:
				self.reporters.complete_extinction()

				# If requested by the user, create a completely new population,
				# otherwise raise an exception.
				if self.config.reset_on_extinction:
					self.population = self.reproduction.create_new(self.config.genome_type,
																   self.config.genome_config,
																   self.config.pop_size)
				else:
					raise CompleteExtinctionException()

			# Divide the new population into species.
			self.species.speciate(self.config, self.population, self.generation)

			self.reporters.end_generation(self.config, self.population, self.species)

			self.generation += 1
			best_genome_generations += 1

		if self.config.no_fitness_termination:
			self.reporters.found_solution(self.config, self.generation, self.best_genome)

		return self.best_genome

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
		p = SmartPopulation(config)

		# Add a stdout reporter to show progress in the terminal.
		p.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		p.add_reporter(stats)
		p.add_reporter(neat.Checkpointer(5, filename_prefix='neat-checkpoint-{}-'.format(self.parameters['enemies'][0])))

		# Run for up to x generations.
		winner = p.run(self.eval_genomes, self.parameters['generations_while_not_improving'])

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