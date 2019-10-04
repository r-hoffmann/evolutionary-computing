import os, pickle#, visualize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure

class PlotResults:
    def __init__(self, algorithms=["GA", "Island", "NEAT"], enemies=[5,6,7]):
        self.algorithms = algorithms
        self.enemies = enemies
        sns.set()
        final_dict = {}
        for algorithm in algorithms:
            for i in [5,6,7]:
                best_fitness_list = []
                mean_fitness_list = []
                for j in range(1,11):
                    final_fitness_list = self.get_last_generation_data(algorithm, i, j)
                    
                    best_fitness_list.append(max(final_fitness_list))
                    mean_fitness_list.append(np.mean(final_fitness_list))
                final_dict[str(algorithm)+str(i)+"_best"] = best_fitness_list
                final_dict[str(algorithm)+str(i)+"_mean"] = mean_fitness_list

        plt.plot()
        plt.scatter([5 for i in range(10)], final_dict["GA5_best"], s = 15 )
        plt.scatter([6 for i in range(10)], final_dict["GA6_best"], s = 15 )
        plt.scatter([7 for i in range(10)], final_dict["GA7_best"], s = 15 )
        plt.scatter([5 for i in range(10)], final_dict["Island5_best"], s = 15 )
        plt.scatter([6 for i in range(10)], final_dict["Island6_best"], s = 15 )
        plt.scatter([7 for i in range(10)], final_dict["Island7_best"], s = 15 )
        plt.scatter([5 for i in range(10)], final_dict["NEAT5_best"], s = 15 )
        plt.scatter([6 for i in range(10)], final_dict["NEAT6_best"], s = 15 )
        plt.scatter([7 for i in range(10)], final_dict["NEAT7_best"], s = 15 )
        plt.xticks([5,6,7])
        #plt.show()
        plt.close()

        all_evals = []
        for algorithm in algorithms:
            mean_best_fitness = []
            std_best_fitness = []
            mean_mean_fitness = []
            std_mean_fitness = []

            for i in [5,6,7]:
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

            all_evals.append(mean_best_fitness)
            all_evals.append(std_best_fitness)
            all_evals.append(mean_mean_fitness)
            all_evals.append(std_mean_fitness)

        #figure(figsize=(2, 3))

        cheat_solution = -.3
        for i, algorithm in enumerate(algorithms):
            df = pd.DataFrame({
                            "enemy": [5,6,7],
                            "algorithm": algorithm, 
                            "best":all_evals[i*2], 
                            "sd":all_evals[i*2+1]})
            #print(df['enemy']+.1)
            plt.errorbar(df['enemy']+cheat_solution, df['best'], yerr=df['sd'], ls='None', marker='o', label=algorithm)
            cheat_solution += .3
        ax = plt.gca()
        ax.xaxis.set_ticks([5,6,7])
        plt.xlabel("enemy")
        plt.ylabel("best fitness")
        plt.legend()
        plt.show()

        #figure(figsize=(2, 3))
        cheat_solution = -.3
        for i, algorithm in enumerate(algorithms):
            df = pd.DataFrame({
                            "enemy": [5,6,7],
                            "algorithm": algorithm, 
                            "mean":all_evals[i*2+2], 
                            "sd":all_evals[i*2+3]})
            plt.errorbar(df['enemy']+cheat_solution, df['mean'], yerr=df['sd'], ls='None', marker='o', label=algorithm)
            cheat_solution += .3
        ax = plt.gca()
        ax.xaxis.set_ticks([5,6,7])
        plt.xlabel("enemy")
        plt.ylabel("mean fitness")
        plt.legend()
        plt.show()

    def get_all_data(self, algorithm, enemy, trial):
        if algorithm=='NEAT':
            final_fitness_list = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))
        elif algorithm=='Island':
            fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))            
            final_fitness_list = [k[0] for k in fitness_row for fitness_row in fitness_record]
        else:
            fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/fitness_record_{}_enemy{}_run{}.pickle".format(algorithm, enemy, trial, algorithm, enemy, trial)), "rb"))
            final_fitness_list = [k[0] for k in fitness_row for fitness_row in fitness_record]
        return final_fitness_list
    
    def get_last_generation_data(self, algorithm, enemy, trial):
        if algorithm=='NEAT':
            final_fitness_list = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))[-1]
        elif algorithm=='Island':
            fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/all_fitnesses.pkl".format(algorithm, enemy, trial)), "rb"))            
            final_fitness_list = [k[0] for k in fitness_record[-1]]
        else:
            fitness_record = pickle.load(open(os.path.join(os.path.dirname(__file__), "../{}_{}_{}/fitness_record_{}_enemy{}_run{}.pickle".format(algorithm, enemy, trial, algorithm, enemy, trial)), "rb"))
            final_fitness_list = [k[0] for k in fitness_record[-1]]
        return final_fitness_list

    def generate_combined_visualize_plots(self, algorithm, enemy, trial):

        for algorithm in self.algorithms:
            for enemy in self.enemies:
                combined_data = []
                for trial in range(1, 11):
                    trial_data = self.get_all_data(algorithm, enemy, trial) 
                    combined_data.append(trial_data)

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
        plt.savefig(os.path.join(os.path.dirname(__file__), "../combined_plots/{}_{}.png".format(algorithm, enemy)))
        plt.close()
        #plt.show()
        #if algorithm=='NEAT':
        #    visualize.plot_species(stats, view=False, filename='{}_{}_{}/speciation.svg'.format(algorithm, enemy, trial))
            # visualize.draw_net(config, winner, True, node_names=node_names, filename="{}/DiGraph".format(self.experiment_name))