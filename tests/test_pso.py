import numpy as np
from core.pso import PSO
from core.bounds import ClampBoundary
from parallel.evaluators import SequentialEvaluator
from objectives.benchmarks import sphere

def test_reproducibility():
    bounds = np.array([[-5.12, 5.12]] * 5)
    
    np.random.seed(42)
    pso_a = PSO(sphere, SequentialEvaluator(), dim=5, bounds=bounds, num_particles=20, w=0.5, c1=1.5, c2=1.5, max_iter=20)
    best_pos_a, best_fit_a, _ = pso_a.optimize()
    
    np.random.seed(42)
    pso_b = PSO(sphere, SequentialEvaluator(), dim=5, bounds=bounds, num_particles=20, w=0.5, c1=1.5, c2=1.5, max_iter=20)
    best_pos_b, best_fit_b, _ = pso_b.optimize()
    
    assert best_fit_a == best_fit_b, "Los fitness óptimos deben coincidir con la misma semilla."
    np.testing.assert_array_equal(best_pos_a, best_pos_b, "Las posiciones óptimas deben coincidir con la misma semilla.")

def test_bounds_handling():
    bounds = np.array([[-2.0, 2.0]] * 3)
    np.random.seed(10)
    pso = PSO(sphere, SequentialEvaluator(), dim=3, bounds=bounds, num_particles=50, w=0.9, c1=2.5, c2=2.5, max_iter=50)
    
    pso.optimize()
    
    for p in pso.particles:
        assert np.all(p.position >= bounds[:, 0]), "Una partícula rompió el límite inferior."
        assert np.all(p.position <= bounds[:, 1]), "Una partícula rompió el límite superior."

def test_monotonic_improvement():
    bounds = np.array([[-5.12, 5.12]] * 5)
    np.random.seed(20)
    pso = PSO(sphere, SequentialEvaluator(), dim=5, bounds=bounds, num_particles=30, w=0.5, c1=1.5, c2=1.5, max_iter=30)
    
    _, _, history = pso.optimize()
    
    for i in range(len(history) - 1):
        assert history[i+1] <= history[i], f"El fitness ha empeorado en la iteración {i}: {history[i]} -> {history[i+1]}"

def test_correctness_sphere():
    bounds = np.array([[-5.12, 5.12]] * 2) # 2D es fácil de optimizar
    np.random.seed(99)
    pso = PSO(sphere, SequentialEvaluator(), dim=2, bounds=bounds, num_particles=50, w=0.7, c1=1.5, c2=1.5, max_iter=150)
    
    best_pos, best_fit, _ = pso.optimize()
    
    assert best_fit < 1e-4, f"PSO ha fallado en converger en Sphere. Fitness final: {best_fit}"