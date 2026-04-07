import numpy as np

class Particle:
    def __init__(self, bounds, dim):
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
        
        v_min = -np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1
        v_max = np.abs(bounds[:, 1] - bounds[:, 0]) * 0.1
        self.velocity = np.random.uniform(v_min, v_max, dim)
        
        self.best_position = np.copy(self.position)
        self.best_fitness = np.inf
        
        self.current_fitness = np.inf