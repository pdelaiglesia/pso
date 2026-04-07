import os
import time
import numpy as np
from core.pso import PSO
from objectives.benchmarks import sphere, sphere_vectorized
from objectives.async_bench import async_sphere
from parallel.evaluators import (SequentialEvaluator, ThreadEvaluator, 
                                 ProcessEvaluator, AsyncEvaluator, 
                                 VectorizedEvaluator, JoblibEvaluator)
from utils.persistence import save_experiment_results

def main():
    os.makedirs("results/benchmarks", exist_ok=True)
    
    dimensions = [2, 10, 30]
    seeds = [42, 100, 2026, 999, 1234] 
    num_particles = 50
    max_iter = 100
    
    evaluators = {
        "V0_Secuencial": (SequentialEvaluator(), sphere),
        "V1_Hilos": (ThreadEvaluator(max_workers=4), sphere),
        "V2_Procesos": (ProcessEvaluator(max_workers=4, chunksize=10), sphere),
        "V3_Asyncio": (AsyncEvaluator(), async_sphere),
        "V4_Vectorizada": (VectorizedEvaluator(), sphere_vectorized),
        "V5_Joblib": (JoblibEvaluator(n_jobs=4), sphere)
    }
    
    print("Iniciando Suite de Benchmarks para PSO\n")
    for dim in dimensions:
        bounds = np.array([[-5.12, 5.12]] * dim)
        print(f"\nEvaluando Dimensión: {dim}D\n ")
        
        for name, (evaluator, func) in evaluators.items():
            print(f"-> Estrategia: {name}")
            
            for seed in seeds:
                np.random.seed(seed)
                
                pso = PSO(
                    objective_func=func, evaluator=evaluator, dim=dim, bounds=bounds, 
                    num_particles=num_particles, w=0.5, c1=1.5, c2=1.5, max_iter=max_iter
                )
                
                start_time = time.perf_counter()
                best_pos, best_fit, history_fitness, _, times_inst = pso.optimize()
                total_time = time.perf_counter() - start_time
                
                config = {
                    "strategy": name, "dim": dim, "seed": seed, "particles": num_particles, 
                    "max_iter": max_iter, "w": 0.5, "c1": 1.5, "c2": 1.5, "cores": 4
                }
                times_inst["total_time"] = total_time
                
                filepath = f"results/benchmarks/{name}_d{dim}_s{seed}.json"
                save_experiment_results(filepath, config, best_fit, history_fitness, times_inst)
            
            print(f"   Completadas 5 semillas para {name} en {dim}.")

if __name__ == "__main__":
    main()