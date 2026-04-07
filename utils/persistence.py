import json
import time

def save_experiment_results(filepath, config, best_fitness, history, times):
    data = {
        "timestamp": time.time(),
        "config": config,
        "results": {
            "best_fitness": best_fitness,
            "history": history,
            "times_instrumentation": times
        }
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)