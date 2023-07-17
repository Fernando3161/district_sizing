import os
from os.path import join


def get_project_root():
    """Return the path to the project root directory.

    :return: A directory path.
    :rtype: str
    """
    return os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    ))


def check_and_create_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder '{folder}' created.")
        else:
            pass
            #print(f"Folder '{folder}' already exists.")


BASE_DIR = get_project_root()
SOURCE_DIR = join(BASE_DIR, "src")
DATA_DIR = join(BASE_DIR,"data")
RESULTS_DIR = join(BASE_DIR,"results")
RESULTS_OPT_DIR = join(RESULTS_DIR,"optimization")
RESULTS_YEAR_DIR = join(RESULTS_DIR,"year")
RESULTS_EXAMPLES_DIR = join(RESULTS_DIR,"district_sizing_examples")
RESULTS_ITERATIONS_DIR = join(RESULTS_DIR, "iterations")
RESULTS_FIGURES_DIR= join(RESULTS_DIR, "figures")

def check_and_create_all_folders():
    folders_to_check = [
        BASE_DIR,
        SOURCE_DIR,
        DATA_DIR,
        RESULTS_DIR,
        RESULTS_OPT_DIR,
        RESULTS_YEAR_DIR,
        RESULTS_EXAMPLES_DIR,
        RESULTS_ITERATIONS_DIR,
        RESULTS_FIGURES_DIR
    ]
    check_and_create_folders(folders_to_check)

if __name__=="__main__":
    check_and_create_all_folders()
    print(BASE_DIR)
    print(RESULTS_OPT_DIR)