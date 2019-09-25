import os
from Framework.Experiment import Experiment

experiment_name = 'experiment_name'
algorithm = 'GA'

parameters = {
    'enemies': [1],
    'config_file': os.path.join(os.path.dirname(__file__), 'NEAT', 'config-NEAT'),
    'generations': 5
}

e = Experiment(experiment_name, algorithm, parameters)