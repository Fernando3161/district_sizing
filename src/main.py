import os
from common import check_and_create_all_folders, RESULTS_DIR, check_and_create_folders
from evaluation import evaluate_results
from optimization import optimize_es


def main(**kwargs):
    """Main function to optimize the energy system and evaluate the results.

    Args:
        kwargs (dict): Keyword arguments containing the configuration parameters.

    Configuration Parameters:
        n_indiv (int): Number of individuals (default: 5).
        n_gen (int): Number of generations (default: 5).
        days (int): Number of days (default: 14).
        start_day (int): Starting day (default: 90).
        n_obj (int): Number of objectives (default: 2).
    """

    n_indiv = kwargs.get("n_indiv", 5)
    n_gen = kwargs.get("n_gen", 5)
    days = kwargs.get("days", 14)
    start_day = kwargs.get("start_day", 90)
    n_obj = kwargs.get("n_obj", 2)
    
    results_folder = f"results_{n_obj}o{n_indiv}i{n_gen}g{days}d"
    results_filename = f"results_{n_obj}o{n_indiv}i{n_gen}g{days}d.csv"

    check_and_create_all_folders()
    output_folder = os.path.join(RESULTS_DIR, results_folder)
    check_and_create_folders([output_folder])

    # Optimize the energy system using the example configuration
    optimize_es(**kwargs)
    evaluate_results(filename=results_filename, output_folder=output_folder)


if __name__ == "__main__":
    # Example configuration
    config = {
        "n_indiv": 50,
        "n_gen": 50,
        "days": 7,
        "start_day": 90,
        "n_obj": 2,
    }

    main(**config)
