import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#algorithms = ["GA", "Island", "NEAT"]
algorithms = ["GA", "NEAT"]

for i in [5,6,7]:
	for j in range(1,11):
		pass

final_dict = {}
for algorithm in algorithms:
	for i in [5,6,7]:
		best_fitness_list = []
		mean_fitness_list = []
		for j in range(1,11):
			#print(i, j)
			#print("../" + str(algorithm) + "_" + str(i) + "_" + str(j) + "/fitness_record_" + str(algorithm) + "_enemy" + str(i) + "_run" + str(j) + ".pickle")
			if algorithm=='GA':
				fitness_record = pickle.load(open("../" + str(algorithm) + "_" + str(i) + "_" + str(j) + "/fitness_record_" + str(algorithm) + "_enemy" + str(i) + "_run" + str(j) + ".pickle", "rb"))
				final_fitness_list = []
				for k in fitness_record[-1]:
					final_fitness_list.append(k[0])
			else:
				final_fitness_list = pickle.load(open("../{}_{}_{}/all_fitnesses.pkl".format(algorithm, i, j), "rb"))			
			
			best_fitness_list.append(max(final_fitness_list))
			mean_fitness_list.append(np.mean(final_fitness_list))
		final_dict[str(algorithm)+str(i)+"_best"] = best_fitness_list
		final_dict[str(algorithm)+str(i)+"_mean"] = mean_fitness_list
#print(final_dict)
plt.plot()
plt.scatter([0 for i in range(10)], final_dict["GA5_best"], s = 15 )
plt.scatter([1 for i in range(10)], final_dict["GA6_best"], s = 15 )
plt.scatter([2 for i in range(10)], final_dict["GA7_best"], s = 15 )
#plt.scatter([0 for i in range(10)], final_dict["Island5_best"] )
#plt.scatter([1 for i in range(10)], final_dict["Island6_best"] )
#plt.scatter([2 for i in range(10)], final_dict["Island7_best"] )
#plt.scatter([0 for i in range(10)], final_dict["NEAT5_best"] )
#plt.scatter([1 for i in range(10)], final_dict["NEAT6_best"] )
#plt.scatter([2 for i in range(10)], final_dict["NEAT7_best"] )
#plt.show()
plt.close()

id = ["Enemy 5", "Enemy 6", "Enemy 7"]
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
			#print(i, j)
			#print("../" + str(algorithm) + "_" + str(i) + "_" + str(j) + "/fitness_record_" + str(algorithm) + "_enemy" + str(i) + "_run" + str(j) + ".pickle")
			if algorithm=='GA':
				fitness_record = pickle.load(open("../" + str(algorithm) + "_" + str(i) + "_" + str(j) + "/fitness_record_" + str(algorithm) + "_enemy" + str(i) + "_run" + str(j) + ".pickle", "rb"))
				final_fitness_list = []
				for k in fitness_record[-1]:
					final_fitness_list.append(k[0])
			else:
				final_fitness_list = pickle.load(open("../{}_{}_{}/all_fitnesses.pkl".format(algorithm, i, j), "rb"))	

			best_fitness_list.append(max(final_fitness_list))
			mean_fitness_list.append(np.mean(final_fitness_list))
		mean_best_fitness.append(np.mean(best_fitness_list))
		std_best_fitness.append(np.std(best_fitness_list))
		mean_mean_fitness.append(np.mean(mean_fitness_list))
		std_mean_fitness.append(np.std(mean_fitness_list))

		#print(np.std(final_fitness_list))
	all_evals.append(mean_best_fitness)
	all_evals.append(std_best_fitness)
	all_evals.append(mean_mean_fitness)
	all_evals.append(std_mean_fitness)

for i in range(len(algorithms)):
	df = pd.DataFrame({"id": id, 
	                   "mean":all_evals[i*2], 
	                   "sd":all_evals[i*2+1]})
	plt.errorbar(np.arange(len(df['id'])), df['mean'], yerr=df['sd'], ls='None', marker='o')
ax = plt.gca()
ax.xaxis.set_ticks(np.arange(len(df['id'])))
ax.xaxis.set_ticklabels(df['id'])
plt.xlabel("id")
plt.ylabel("mean")
plt.show()

for i in range(len(algorithms)):
	df = pd.DataFrame({"id": id, 
	                   "mean":all_evals[i*2+2], 
	                   "sd":all_evals[i*2+3]})
	plt.errorbar(np.arange(len(df['id'])), df['mean'], yerr=df['sd'], ls='None', marker='o')
ax = plt.gca()
ax.xaxis.set_ticks(np.arange(len(df['id'])))
ax.xaxis.set_ticklabels(df['id'])
plt.xlabel("id")
plt.ylabel("mean")
plt.show()