import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_convergence(history, title="Convergencia del PSO", filename=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Mejor Fitness Global', color='blue', linewidth=2)
    plt.yscale('log') 
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def animate_pso_2d(func, bounds, pos_history, title="Evolución PSO 2D", filename="pso_anim.gif"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(bounds[0, 0], bounds[0, 1], 100)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
            
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    fig.colorbar(contour, ax=ax, label='Fitness')
    
    scatter = ax.scatter([], [], c='red', edgecolors='white', s=50, label='Partículas')
    best_point = ax.scatter([], [], c='gold', marker='*', edgecolors='black', s=200, label='Mejor Global')
    
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    title_text = ax.set_title(f'{title} - Iteración: 0')
    ax.legend(loc="upper right")
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        best_point.set_offsets(np.empty((0, 2)))
        return scatter, best_point
        
    def update(frame):
        positions = pos_history[frame]
        scatter.set_offsets(positions)
        
        best_idx = np.argmin([func(p) for p in positions])
        best_point.set_offsets([positions[best_idx]])
        
        title_text.set_text(f'{title} - Iteración: {frame}')
        return scatter, best_point, title_text
        
    anim = animation.FuncAnimation(fig, update, frames=len(pos_history), init_func=init, blit=True, interval=150)
    
    anim.save(filename, writer='pillow', fps=10)
    plt.close()