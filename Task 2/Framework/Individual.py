import numpy as np

class Individual():
    def __init__(self):
        self.network = []
        self.fitness = [] # should be easiest if assigned here, but I can't
        self.mutation_rate = 0
