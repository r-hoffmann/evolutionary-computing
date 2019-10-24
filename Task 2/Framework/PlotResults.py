import os, pickle#, visualize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure

class PlotResults:
	def __init__(self, algorithms=["GA_package", "NEAT", "Island"], enemies=[[1,3,6]]):
		self.algorithms = algorithms
		self.enemies = enemies
		sns.set()
		self.generate_combined_fitness_plots()
		# self.other_graphs()

	def other_graphs(self):
		all_evals = []
		for algorithm in self.algorithms:
			mean_best_fitness = []
			std_best_fitness = []
			mean_mean_fitness = []
			std_mean_fitness = []
			for i in self.enemies:
				best_fitness_list = []
				mean_fitness_list = []
				for j in range(1,11):
					final_fitness_list = self.get_last_generation_data(algorithm, i, j)
					best_fitness_list.append(max(final_fitness_list))
					mean_fitness_list.append(np.mean(final_fitness_list))
				mean_best_fitness.append(np.mean(best_fitness_list))
				std_best_fitness.append(np.std(best_fitness_list))
				mean_mean_fitness.append(np.mean(mean_fitness_list))
				std_mean_fitness.append(np.std(mean_fitness_list))
				print('the mean best fitness of algorithm %s for enemy %s is %f' % (algorithm,i,np.mean(best_fitness_list)))

			all_evals.append(mean_best_fitness)
			all_evals.append(std_best_fitness)
			all_evals.append(mean_mean_fitness)
			all_evals.append(std_mean_fitness)
		#figure(figsize=(2, 3))

		cheat_solution = -.3
		for i, algorithm in enumerate(self.algorithms):
			df = pd.DataFrame({
							"enemy": str(self.enemies),
							"algorithm": algorithm, 
							"best":all_evals[i*4], 
							"sd":all_evals[i*4+1]})
			#print(df['enemy']+.1)
			print(all_evals[i*2])
			plt.errorbar(df['enemy']+cheat_solution, df['best'], yerr=df['sd'], ls='None', marker='o', label=algorithm)
			cheat_solution += .3
		ax = plt.gca()
		ax.xaxis.set_ticks(self.enemies)
		plt.xlabel("enemy")
		plt.ylabel("best fitness")
		plt.legend()
		plt.savefig(os.path.join(os.path.dirname(__file__), "../combined_plots/algorithms_best.png"))
		plt.show()

		#figure(figsize=(2, 3))
		cheat_solution = -.3
		for i, algorithm in enumerate(self.algorithms):
			df = pd.DataFrame({
							"enemy": self.enemies,
							"algorithm": algorithm, 
							"mean":all_evals[i*4+2], 
							"sd":all_evals[i*4+3]})
			plt.errorbar(df['enemy']+cheat_solution, df['mean'], yerr=df['sd'], ls='None', marker='o', label=algorithm)
			cheat_solution += .3
		ax = plt.gca()
		ax.xaxis.set_ticks(self.enemies)
		plt.xlabel("enemy")
		plt.ylabel("mean fitness")
		plt.legend()
		plt.savefig(os.path.join(os.path.dirname(__file__), "../combined_plots/algorithms_mean.png"))
		plt.show()

	def get_all_data(self, algorithm, enemy, trial):
		if algorithm=='Island':
			fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))			
			final_fitness_list = [[k[0] for k in fitness_row] for fitness_row in fitness_record]
		else:
			final_fitness_list = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))
		return final_fitness_list
	
	def get_last_generation_data(self, algorithm, enemy, trial):
		if algorithm=='Island':
			fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))			
			final_fitness_list = [k[0] for k in fitness_record[-1]]
		else:
			final_fitness_list = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))[-1]
		return final_fitness_list

	def generate_combined_fitness_plots(self):
		algorithm_colors = sns.color_palette()[:3]
		for enemy in self.enemies:
			for algorithm, algorithm_color in zip(self.algorithms, algorithm_colors):
				combined_data = []
				for generation in range(1, 51):
					combined_data.append([])

				for trial in range(1, 11):
					trial_data = self.get_all_data(algorithm, enemy, trial) 
					for i, generation in enumerate(trial_data):
						if algorithm=='Island':
							if i%3==0:
								combined_data[i//3] += [g/8 for g in generation]
						else:
							if i < 50:
								combined_data[i] += generation
				mean_data = []
				std_minus_data = []
				std_plus_data = []
				best_data = []
				for generation in combined_data:
					best_data.append(np.max(generation))
					calculated_mean = np.mean(generation)
					calculated_std = np.std(generation)
					mean_data.append(calculated_mean)
					std_minus_data.append(calculated_mean - calculated_std)
					std_plus_data.append(calculated_mean + calculated_std)

				plt.plot(mean_data, label=algorithm)
				plt.plot(std_minus_data, linestyle='--', color=algorithm_color)
				plt.plot(std_plus_data, linestyle='--', color=algorithm_color)
				plt.plot(best_data, linestyle=':', color=algorithm_color)
			plt.legend()
			plt.xlabel('Fitness evaluation')
			plt.ylabel('Fitness score')
			plt.savefig(os.path.join(os.path.dirname(__file__), "../combined_plots/enemy_{}.png".format(enemy)))
			plt.close()
