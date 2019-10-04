import pickle
import numpy as np

pickle_in = open('TestStochasticity_7_1/fitness_record_enemy7_run1.pickle','rb')
one_individual = pickle.load(pickle_in)
pickle_in.close()
print('this is how one_individual looks:\n',one_individual)

multiple_individuals = [[[12.244257786413087, 0.0, 20.0, 316.0, 18.0], [21.216174817670264, 0.0, 30.0, 325.0, 27.0], [21.228558876869982, 0.0, 30.0, 321.0, 27.0], [12.260207087820767, 0.0, 20.0, 311.0, 18.0], [21.159358342626604, 0.0, 30.0, 344.0, 27.0], [12.260207087820767, 0.0, 20.0, 311.0, 18.0], [12.065105804380412, 0.0, 20.0, 378.0, 18.0], [21.17399989261955, 0.0, 30.0, 339.0, 27.0], [12.038994660376726, 0.0, 20.0, 388.0, 18.0], [12.171054382389793, 0.0, 20.0, 340.0, 18.0], [39.33357331188757, 0.0, 50.0, 289.0, 45.0], [21.228558876869982, 0.0, 30.0, 321.0, 27.0], [21.3062678611973, 0.0, 30.0, 297.0, 27.0], [21.216174817670264, 0.0, 30.0, 325.0, 27.0], [1.0, 0.0, 0.0, 314.0, 0.0], [12.194865031083513, 0.0, 20.0, 332.0, 18.0], [12.247427361174367, 0.0, 20.0, 315.0, 18.0], [21.12506926914797, 0.0, 30.0, 356.0, 27.0], [12.049357447412273, 0.0, 20.0, 384.0, 18.0], [21.22544845445559, 0.0, 30.0, 322.0, 27.0]]]
print('this is how multiple_inividuals looks:\n',multiple_individuals)

fitnesses_one = []
for eval in one_individual:
    fitnesses_one.append(eval[0][0])
print('fitnesses_one=\n',fitnesses_one)

fitnesses_multiple = []
for indi in multiple_individuals[0] :
    fitnesses_multiple.append(indi[0])
print('fitnesses_multiple=\n',fitnesses_multiple)

print('the mean of one individual is %f and the std is %f' % (np.mean(fitnesses_one),np.std(fitnesses_one)))
print('the mean of twenty individuals is %f and the std is %f' % (np.mean(fitnesses_multiple),np.std(fitnesses_multiple)))