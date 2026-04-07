import numpy as np
import os
import json
from experiments.grid_search import run_grid_search
from objectives.benchmarks import sphere
from parallel.evaluators import SequentialEvaluator

def main():
    os.makedirs("results/grid_search", exist_ok=True)
    
    dim = 10
    bounds = np.array([[-5.12, 5.12]] * dim)
    
    param_grid = {
        'num_particles': [30],
        'w': [0.5, 0.7, 0.9],
        'c1': [1.0, 1.5, 2.0],
        'c2': [1.0, 1.5, 2.0]
    }
    
    print("Iniciando Grid Search para PSO en función Sphere 10D")
    evaluator = SequentialEvaluator()
    
    results = run_grid_search(
        objective_func=sphere, 
        evaluator=evaluator, 
        dim=dim, 
        bounds=bounds, 
        param_grid=param_grid, 
        max_iter=50, 
        seeds=[42, 100, 200, 300, 400]
    )
    
    print("\n=== TOP 3 ===")
    for i in range(min(3, len(results))):
        res = results[i]
        print(f"{i+1}. Fitness Promedio: {res['avg_fitness']:.6e} | Parámetros: {res['params']}")
        
    filepath = "results/grid_search/best_grid_results.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados completos guardados en {filepath}")

if __name__ == "__main__":
    main()