import os

from Framework.GeneticAlgorithm import GeneticAlgorithm
from Framework.NeatAlgorithm import NeatAlgorithm
from Framework.Framework import Framework
from Framework.PlayerNeatController import PlayerNeatController

class Experiment:
    def __init__(self, experiment_name, algorithm, parameters=dict()):
        self.experiment_name = experiment_name
        self.parameters = parameters
        self.parameters['experiment_name'] = experiment_name
        
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        if algorithm=='GA':
            self.parameters['player_controller'] = None
            self.algorithm = GeneticAlgorithm(parameters)
        elif algorithm=='NEAT':
            self.parameters['player_controller'] = PlayerNeatController()
            self.algorithm = NeatAlgorithm(parameters)
        else:
            raise ValueError('{} is not an algorithm'.format(algorithm))
        

        self.algorithm.run()