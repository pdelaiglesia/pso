import numpy as np
import time
import logging
from core.particle import Particle
from core.bounds import ClampBoundary
from core.topology import GlobalBestTopology

logger = logging.getLogger("PSO")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [Iter %(iteration)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False 

class PSO:
    def __init__(self, objective_func, evaluator, dim, bounds, num_particles, w, c1, c2, max_iter, tol=1e-6, patience=15, topology=None):
        self.objective_func = objective_func
        self.evaluator = evaluator
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        
        self.boundary_policy = ClampBoundary()
        self.topology = topology if topology else GlobalBestTopology()
        
        self.particles = [Particle(bounds, dim) for _ in range(num_particles)]
        self.global_best_pos = np.zeros(dim)
        self.global_best_fitness = np.inf

    def optimize(self):
        history_fitness = []
        history_positions = []
        
        times_instrumentation = {
            "eval_time": 0.0,
            "update_time": 0.0
        }
        
        stagnation_counter = 0
        last_best_fitness = np.inf
        
        logger.info(f"Iniciando optimización con {self.num_particles} partículas ({type(self.evaluator).__name__})", extra={"iteration": 0})

        for iteration in range(self.max_iter):
            t0_eval = time.perf_counter()
            positions = [p.position for p in self.particles]
            fitness_values = self.evaluator.evaluate(self.objective_func, positions)
            times_instrumentation["eval_time"] += (time.perf_counter() - t0_eval)
            
            t0_update = time.perf_counter()
            for i, p in enumerate(self.particles):
                p.current_fitness = fitness_values[i]
                if p.current_fitness < p.best_fitness:
                    p.best_fitness = p.current_fitness
                    p.best_position = np.copy(p.position)
            
            self.global_best_pos, self.global_best_fitness = self.topology.update_global_best(
                self.particles, self.global_best_pos, self.global_best_fitness
            )
            
            for p in self.particles:
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (p.best_position - p.position)
                social = self.c2 * r2 * (self.global_best_pos - p.position)
                p.velocity = self.w * p.velocity + cognitive + social
                p.position = p.position + p.velocity
                p.position, p.velocity = self.boundary_policy.apply(p.position, p.velocity, self.bounds)
                
            times_instrumentation["update_time"] += (time.perf_counter() - t0_update)

            history_fitness.append(self.global_best_fitness)
            history_positions.append(np.array([p.position for p in self.particles]))
            
            if iteration % 20 == 0 or iteration == self.max_iter - 1:
                logger.info(f"Mejor fitness por ahora: {self.global_best_fitness:.6e}", extra={"iteration": iteration})

            if abs(last_best_fitness - self.global_best_fitness) < self.tol:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_best_fitness = self.global_best_fitness
                
            if stagnation_counter >= self.patience:
                logger.warning(f"Estancamiento. Llegamos a la tolerancia: {self.tol} ", extra={"iteration": iteration})
                break
                
        logger.info("FIN", extra={"iteration": iteration})
            
        return self.global_best_pos, self.global_best_fitness, history_fitness, history_positions, times_instrumentation