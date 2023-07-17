import os
import logging
from time import time
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from common import RESULTS_OPT_DIR
from problem import ProblemES

logging.basicConfig(level=logging.ERROR)


def optimize_es(**kwargs):
    """
    Optimize the energy system using multi-objective optimization.

    Parameters:
        **kwargs (dict): Keyword arguments for configuring the optimization.
            - n_indiv (int): Number of individuals in the population. Default is 5.
            - n_gen (int): Number of generations for optimization. Default is 5.
            - days (int): Number of days for optimization. Default is 14.
            - start_day (int): Starting day for optimization. Default is 90.
            - n_obj (int): Number of objectives for optimization. Default is 2.
            - filename (str): Filename for saving the optimization results. Default is 'results_<n_obj>o<n_indiv>i<n_gen>g.csv'.

    Returns:
        F (numpy.ndarray): The optimized objective values.

    """
    N_INDIV = kwargs.get("n_indiv", 5)
    N_GEN = kwargs.get("n_gen", 5)
    DAYS = kwargs.get("days", 14)
    START_DAY = kwargs.get("start_day", 90)
    N_OBJ = kwargs.get("n_obj", 2)
    RESULTS_FILENAME = f"results_{N_OBJ}o{N_INDIV}i{N_GEN}g{DAYS}d.csv"


    problem = ProblemES()
    problem.days = DAYS
    problem.start_day = START_DAY
    problem.n_obj = N_OBJ
    problem.n_indivs = N_INDIV
    algorithm = NSGA2(pop_size=N_INDIV)
    res = minimize(problem, algorithm, termination=("n_gen", N_GEN), seed=1)
    problem.finalize(filename=os.path.join(RESULTS_OPT_DIR, RESULTS_FILENAME))

    return res.F


if __name__ == "__main__":
    # Example configuration
    config = {
        "n_indiv": 10,
        "n_gen": 10,
        "days": 7,
        "start_day": 90,
        "n_obj": 2,
        #"filename": "test_foo.csv"
    }

    # Optimize the energy system using the example configuration
    optimized_values = optimize_es(**config)
    print("Optimized Objective Values:")
    print(optimized_values)
