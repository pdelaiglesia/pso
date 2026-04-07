import numpy as np
from core.pso import PSO
from parallel.evaluators import SequentialEvaluator
from objectives.benchmarks import sphere, rosenbrock
from viz.plot import animate_pso_2d

def main():
    dim = 2
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    
    print("Iniciando ejecución de PSO en función Sphere 2D...")
    np.random.seed(42)
    evaluator = SequentialEvaluator()
    
    pso = PSO(
        objective_func=sphere, 
        evaluator=evaluator, 
        dim=dim, 
        bounds=bounds, 
        num_particles=25, 
        w=0.6, c1=1.5, c2=1.5, max_iter=50
    )
    
    best_pos, best_fit, history_fitness, history_positions, internal_times = pso.optimize()
    
    print(f"PSO Finalizado. Mejor Fitness: {best_fit:.6f}")
    print("Generando archivo GIF")
    
    animate_pso_2d(
        func=sphere, 
        bounds=bounds, 
        pos_history=history_positions, 
        title="PSO Sphere 2D",
        filename="pso_sphere_2d.gif"
    )

if __name__ == "__main__":
    main()