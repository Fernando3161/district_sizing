'''
Main file for simulation
'''

from src.es_optimize import optimize_es_size


config = {
        "n_indiv": 5,
        "n_gen": 2,
        "days": 7,
        "start_day": 90,
        "n_obj": 2,
    }

if __name__ == "__main__":
    optimize_es_size(**config)
    

    