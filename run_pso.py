import time
import numpy as np
from core.pso import PSO
from objectives.benchmarks import sphere
from parallel.evaluators import SequentialEvaluator
from utils.config import load_config

def main():
    config = load_config()
    print(f"\n--- Ejecutando PSO con configuración ---")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("-" * 40 + "\n")
    
    dim = config['dim']
    bounds = np.array([[-5.12, 5.12] for _ in range(dim)]) 
    
    np.random.seed(42)
    evaluator_seq = SequentialEvaluator()
    
    pso = PSO(
        objective_func=sphere, 
        evaluator=evaluator_seq, 
        dim=dim, 
        bounds=bounds, 
        num_particles=config['particles'], 
        w=config['w'], 
        c1=config['c1'], 
        c2=config['c2'], 
        max_iter=config['max_iter']
    )
    
    start_time = time.perf_counter()
    best_pos, best_fit, history_seq, _, internal_times = pso.optimize()
    end_time = time.perf_counter()
    
    print(f"\n=== RESULTADOS FINALES ===")
    print(f"Mejor Fitness: {best_fit:.6e}")
    print(f"Tiempo Total: {end_time - start_time:.4f}s")
    
    t_eval = internal_times['eval_time']
    t_update = internal_times['update_time']
    print(f"  -> Tiempo evaluando fitness: {t_eval:.4f}s")
    print(f"  -> Tiempo actualizando partículas: {t_update:.4f}s")

if __name__ == "__main__":
    main()