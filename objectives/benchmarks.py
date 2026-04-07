
import numpy as np

def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x*x) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)
def sphere_vectorized(X):
    return np.sum(X**2, axis=1)

def rastrigin_vectorized(X):
    A = 10
    return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)