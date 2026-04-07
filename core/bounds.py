import numpy as np

class BoundaryPolicy:
    def apply(self, position, velocity, bounds):
        raise NotImplementedError("Debe ser implementado por subclases")

class ClampBoundary(BoundaryPolicy):
    def apply(self, position, velocity, bounds):
        new_pos = np.copy(position)
        new_vel = np.copy(velocity)
        
        lower_bound_mask = new_pos < bounds[:, 0]
        new_pos[lower_bound_mask] = bounds[lower_bound_mask, 0]
        new_vel[lower_bound_mask] = 0.0
        
        upper_bound_mask = new_pos > bounds[:, 1]
        new_pos[upper_bound_mask] = bounds[upper_bound_mask, 1]
        new_vel[upper_bound_mask] = 0.0
        
        return new_pos, new_vel

class ReflectBoundary(BoundaryPolicy):

    def apply(self, position, velocity, bounds):
        new_pos = np.copy(position)
        new_vel = np.copy(velocity)
        
        out_of_bounds = (new_pos < bounds[:, 0]) | (new_pos > bounds[:, 1])
        new_pos = np.clip(new_pos, bounds[:, 0], bounds[:, 1])
        
        new_vel[out_of_bounds] *= -1.0 
        
        return new_pos, new_vel