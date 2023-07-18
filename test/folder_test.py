import os
import unittest
import sys
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(path)
from src.financial import calculate_annualized_npv
from src.common import *

class TestYourModule(unittest.TestCase):

    def test_get_project_root(self):
        # Test if the project root is a valid directory
        root = get_project_root()
        self.assertTrue(os.path.isdir(root))

    def test_check_and_create_folders(self):
        # Test if the folders are created correctly
        folders = ["test_folder1", "test_folder2"]
        check_and_create_folders(folders)
        self.assertTrue(os.path.exists("test_folder1"))
        self.assertTrue(os.path.exists("test_folder2"))
        os.rmdir("test_folder1")
        os.rmdir("test_folder2")

    def test_check_and_create_all_folders(self):
        # Test if all folders are created correctly
        check_and_create_all_folders()
        self.assertTrue(os.path.exists(BASE_DIR))
        self.assertTrue(os.path.exists(SOURCE_DIR))
        self.assertTrue(os.path.exists(DATA_DIR))
        self.assertTrue(os.path.exists(RESULTS_DIR))
        self.assertTrue(os.path.exists(RESULTS_OPT_DIR))
        self.assertTrue(os.path.exists(RESULTS_YEAR_DIR))
        self.assertTrue(os.path.exists(RESULTS_EXAMPLES_DIR))
        self.assertTrue(os.path.exists(RESULTS_ITERATIONS_DIR))
        self.assertTrue(os.path.exists(RESULTS_FIGURES_DIR))
        self.assertFalse(os.path.exists(RESULTS_FIGURES_DIR+"1"))



if __name__ == "__main__":
    unittest.main()