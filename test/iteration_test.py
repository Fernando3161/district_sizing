import os
import unittest
import sys
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
from src.common import *
from src.scenarios_eval import (
    evaluate_pv_wind_sizes,
    evaluate_plot_iterations,
    evaluate_plot_results,
)

import warnings
warnings.filterwarnings("ignore")
import logging
import pandas as pd
logging.basicConfig(level=logging.ERROR)


class TestYourModule(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test results
        self.test_results_dir = os.path.join(TEST_DIR,"test_results_iteration")
        self.csv_data = None
        os.makedirs(self.test_results_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory after each test
        for file in os.listdir(self.test_results_dir):
            os.remove(os.path.join(self.test_results_dir, file))
        os.rmdir(self.test_results_dir)

        for file in os.listdir(os.path.join(TEST_DIR,"test_data")):
            if "iteration_results" not in str(file):
                os.remove(os.path.join(TEST_DIR,"test_data", file))

    def test_evaluate_pv_wind_sizes(self):
        # Test if the function returns a valid results filename
        results_filename = evaluate_pv_wind_sizes(n_days=7, grid_resolution=100, folder =self.test_results_dir)
        self.assertIsInstance(results_filename, str)
        self.assertTrue(results_filename.endswith(".csv"))
        self.assertTrue(os.path.exists(os.path.join(self.test_results_dir,results_filename)))
        self.csv_data = pd.read_csv(os.path.join(self.test_results_dir,results_filename))

    def test_evaluate_plot_iterations(self):
        # Test if the function creates the plot files successfully
        results_filename = "iteration_results_7d_50kW.csv"
        folder = os.path.join(TEST_DIR, "test_data")

        evaluate_plot_iterations(results_filename=results_filename, folder=folder)
        self.assertTrue(os.path.exists(os.path.join(folder, "solutions_7d_50kW.png")))
        self.assertTrue(os.path.exists(os.path.join(folder, "technologies_7d_50kW.png")))

    def test_evaluate_plot_results(self):
        # Test if the function runs without errors
        evaluate_plot_results(n_days=1, grid_resolution=50, folder=self.test_results_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_results_dir, "iteration_results_1d_50kW.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_results_dir, "solutions_1d_50kW.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_results_dir, "technologies_1d_50kW.png")))


if __name__ == "__main__":
    unittest.main()