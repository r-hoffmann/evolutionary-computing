import os

from Framework.GeneticAlgorithm import GeneticAlgorithm
from Framework.GeneticAlgorithmIndividualClass import GeneticAlgorithmIC
from Framework.NeatAlgorithm import NeatAlgorithm
from Framework.PlayerNeatController import PlayerNeatController
from Framework.IslandAlgorithm import IslandAlgorithm
from Framework.TestStochasticity import TestStochasticity


class Experiment:
    def __init__(self, experiment_name, algorithm, parameters=dict()):
        self.experiment_name = experiment_name
        self.parameters = parameters
        self.parameters['experiment_name'] = experiment_name

        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        if algorithm == 'GA':
            self.parameters['player_controller'] = None
            self.algorithm = GeneticAlgorithmIC(parameters)
        elif algorithm == 'NEAT':
            self.parameters['player_controller'] = PlayerNeatController()
            self.algorithm = NeatAlgorithm(parameters)
        elif algorithm == 'GA_package':
            self.parameters['player_controller'] = PlayerNeatController()
            self.algorithm = NeatAlgorithm(parameters)
        elif algorithm == 'Island':
            self.parameters['player_controller'] = None
            self.algorithm = IslandAlgorithm(parameters)
        elif algorithm == 'TestStochasticity':
            self.parameters['player_controller'] = None
            self.algorithm = TestStochasticity(parameters)
        else:
            raise ValueError('{} is not an algorithm'.format(algorithm))

    def run(self):
        return self.algorithm.run()

    def test(self):
        self.algorithm.test_model = self.parameters['test_model']
        return self.algorithm.test()
