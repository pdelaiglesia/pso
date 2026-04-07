import itertools
import numpy as np
import time
from core.pso import PSO

def run_grid_search(objective_func, evaluator, dim, bounds, param_grid, max_iter=100, seeds=[42, 100, 2026]):
    results = []
    keys, values = zip(*param_grid.items())
    
    print(f"{'Parámetros':<45} | {'Fitness':<10} | {'Iters':<6} | {'Tiempo(s)':<8}")
    print("----------------------------------------------")
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        fitness_across_seeds = []
        times_across_seeds = []
        iters_across_seeds = []
        
        for seed in seeds:
            np.random.seed(seed)
            
            pso = PSO(
                objective_func=objective_func, evaluator=evaluator, dim=dim, bounds=bounds,
                num_particles=params['num_particles'], w=params['w'], 
                c1=params['c1'], c2=params['c2'], max_iter=max_iter
            )
            
            start_time = time.perf_counter()
            best_pos, best_fit, history, _, internal_times = pso.optimize()
            end_time = time.perf_counter()
            
            fitness_across_seeds.append(best_fit)
            times_across_seeds.append(end_time - start_time)
            iters_across_seeds.append(len(history)) # Iteraciones hasta parar
            
        avg_fitness = np.mean(fitness_across_seeds)
        std_fitness = np.std(fitness_across_seeds)
        avg_time = np.mean(times_across_seeds)
        avg_iters = np.mean(iters_across_seeds)
        
        results.append({
            "params": params, "avg_fitness": avg_fitness, "std_fitness": std_fitness,
            "avg_time": avg_time, "avg_iters": avg_iters
        })
        
        param_str = f"w:{params['w']}, c1:{params['c1']}, c2:{params['c2']}, N:{params['num_particles']}"
        print(f"{param_str:<45} | {avg_fitness:<10.6f} | {avg_iters:<6.1f} | {avg_time:<8.4f}")
        
    results.sort(key=lambda x: x['avg_fitness'])
    return results