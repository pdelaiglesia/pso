import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze():
    results_dir = "results/benchmarks"
    if not os.path.exists(results_dir):
        print(f"Error: No existe la carpeta '{results_dir}'.")
        return
        
    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    data = defaultdict(lambda: defaultdict(list))
    
    for f in files:
        with open(os.path.join(results_dir, f), 'r') as file:
            content = json.load(file)
            strategy = content["config"]["strategy"]
            dim = content["config"]["dim"]
            data[dim][strategy].append(content)
            
    if not data:
        print("No hay archivos JSON para analizar en la carpeta.")
        return

    for dim in sorted(data.keys()):
        print(f"\n" + "="*95)
        print(f"ANALIZANDO DIMENSIÓN: {dim}D")
        print("---------------------------------------------------")
        
        print(f"{'Estrategia':<18} | {'Fit Medio':<12} | {'Iters':<6} | {'Tiempo(s)':<10} | {'Speedup':<8} | {'Eficiencia'}")
        print("---------------------------------------------------")
        
        t_baseline = 1.0
        if "V0_Secuencial" in data[dim]:
            t_baseline = np.mean([run["results"]["times_instrumentation"]["total_time"] for run in data[dim]["V0_Secuencial"]])
        
        strategies = sorted(data[dim].keys())
        
        for strategy in strategies:
            runs = data[dim][strategy]

            avg_fit = np.mean([r["results"]["best_fitness"] for r in runs])
            avg_iters = np.mean([len(r["results"]["history"]) for r in runs])
            avg_time = np.mean([r["results"]["times_instrumentation"]["total_time"] for r in runs])
            
            speedup = t_baseline / avg_time if avg_time > 0 else 0
            
            cores = runs[0]["config"].get("cores", 1)
            if strategy in ["V0_Secuencial", "V4_Vectorizada", "V3_Asyncio"]:
                cores = 1
                
            eficiencia = speedup / cores
            
            print(f"{strategy:<18} | {avg_fit:<12.6f} | {avg_iters:<6.1f} | {avg_time:<10.4f} | {speedup:<8.2f}x | {eficiencia:.2f}")
            
        plt.figure(figsize=(10, 6))
        fitness_data = []
        labels = []
        for strategy in strategies:
            fitness_data.append([r["results"]["best_fitness"] for r in data[dim][strategy]])
            labels.append(strategy.split("_")[0])
            
        plt.boxplot(fitness_data, labels=labels, patch_artist=True)
        plt.yscale('log')
        plt.title(f"Distribución del Mejor Fitness Final ({dim}D) - 5 Semillas", fontsize=14)
        plt.ylabel("Fitness (Escala Logarítmica)", fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        boxplot_path = os.path.join(results_dir, f"boxplot_d{dim}.png")
        plt.savefig(boxplot_path)
        plt.close()
        
        plt.figure(figsize=(12, 6))
        for strategy in strategies:
            histories = [r["results"]["history"] for r in data[dim][strategy]]
            

            max_len = max(len(h) for h in histories)
            padded_histories = []
            for h in histories:
                padded = h + [h[-1]] * (max_len - len(h))
                padded_histories.append(padded)
                
            mean_history = np.mean(padded_histories, axis=0)
            plt.plot(mean_history, label=strategy, linewidth=2)
            
        plt.yscale('log')
        plt.title(f"Curvas de Convergencia Promedio ({dim}D)", fontsize=14)
        plt.xlabel("Iteraciones", fontsize=12)
        plt.ylabel("Mejor Fitness Global Promedio (Log)", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        conv_path = os.path.join(results_dir, f"convergencia_d{dim}.png")
        plt.savefig(conv_path)
        plt.close()

    print(f"\nAnálisis finalizado. Guardados en '{results_dir}'")

if __name__ == "__main__":
    analyze()