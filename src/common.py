'''
Created on 20.05.2021

@author: Fernando Penaherrera @UOL/OFFIS
'''
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


BASE_DIR = get_project_root()
SOURCE_DIR = join(BASE_DIR, "src")
DATA_DIR = join(BASE_DIR,"data")
RESULTS_DIR = join(BASE_DIR,"results")



if __name__ == '__main__':
    print(BASE_DIR)

    pass