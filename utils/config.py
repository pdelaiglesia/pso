import argparse
import json
import os

def load_config():
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization (PSO) Runner")
    
    parser.add_argument('--config', type=str, default=None, help="Ruta a un archivo JSON de configuración")

    parser.add_argument('--dim', type=int, default=10, help="Dimensión del problema")
    parser.add_argument('--particles', type=int, default=30, help="Tamaño del enjambre")
    parser.add_argument('--max_iter', type=int, default=100, help="Iteraciones máximas")
    parser.add_argument('--w', type=float, default=0.5, help="Inercia w")
    parser.add_argument('--c1', type=float, default=1.5, help="Coeficiente cognitivo c1 ")
    parser.add_argument('--c2', type=float, default=1.5, help="Coeficiente social c2 ")
    
    args = parser.parse_args()
    
    config = vars(args).copy()
    
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"Configuración cargada desde archivo: {args.config}")
        else:
            print(f"No se ha encontrado el archivo {args.config}.")
            
    return config