import numpy as np

class TopologyStrategy:
    def update_global_best(self, particles, current_global_best_pos, current_global_best_fitness):
        raise NotImplementedError("Debe implementarse en la subclase")

class GlobalBestTopology(TopologyStrategy):
    def update_global_best(self, particles, current_global_best_pos, current_global_best_fitness):
        best_pos = current_global_best_pos
        best_fit = current_global_best_fitness
        
        for p in particles:
            if p.current_fitness < best_fit:
                best_fit = p.current_fitness
                best_pos = np.copy(p.position)
                
        return best_pos, best_fit