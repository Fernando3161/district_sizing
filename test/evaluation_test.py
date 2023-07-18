import os
import unittest
import sys
import warnings
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
import pandas as pd
import os

from src.evaluation import (
    get_pareto_front,
    plot_all_solutions,
    plot_kpi_pairgrid,
    plot_tech_pairgrid,
    plot_pareto,
    evaluate_results
)
from src.common import RESULTS_OPT_DIR

warnings.filterwarnings("ignore")

class YourModuleTest(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.results_pareto = pd.DataFrame({
            'Gen': [1, 1, 2, 2, 3, 3],
            'total_co2': [10, 20, 15, 25, 18, 30],
            'total_system_costs': [100, 150, 120, 180, 130, 200]
        })
        filename= "results_year.csv"

        results =     pd.read_csv(os.path.join(RESULTS_OPT_DIR,filename))
        self.results = results.tail(100)

        # Set up output folder for testing
        self.output_folder = 'test_output'

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def tearDown(self):
        # Clean up the output folder after testing
        plot_files = [
            os.path.join(self.output_folder, 'solutions.png'),
            os.path.join(self.output_folder, 'pairplot_kpi.png'),
            os.path.join(self.output_folder, 'pairplot_techs.png'),
            os.path.join(self.output_folder, 'sizing.png'),
            os.path.join(self.output_folder, 'correl.png')
        ]

        for file in plot_files:
            if os.path.isfile(file):
                os.remove(file)

        os.rmdir(self.output_folder)

    def test_get_pareto_front(self):
        expected_pareto_front = pd.DataFrame({
            'Gen': [1, 2, 3],
            'total_co2': [10, 15, 18],
            'total_system_costs': [100, 120, 130],
            'ParetoOptimal': [True, True, True]
        })

        pareto_front = get_pareto_front(self.results_pareto)
        pd.testing.assert_frame_equal(pareto_front, expected_pareto_front)

    def test_plot_all_solutions(self):
        # Test that the function runs without errors
        plot_all_solutions(self.results, self.output_folder)

        # Check if the plot file exists
        plot_file = os.path.join(self.output_folder, 'solutions.png')
        self.assertTrue(os.path.isfile(plot_file))

    def test_plot_kpi_pairgrid(self):
        # Test that the function runs without errors
        plot_kpi_pairgrid(self.results, self.output_folder)

        # Check if the plot file exists
        plot_file = os.path.join(self.output_folder, 'pairplot_kpi.png')
        self.assertTrue(os.path.isfile(plot_file))

    def test_plot_tech_pairgrid(self):
        # Test that the function runs without errors
        plot_tech_pairgrid(self.results, self.output_folder)

        # Check if the plot file exists
        plot_file = os.path.join(self.output_folder, 'pairplot_techs.png')
        self.assertTrue(os.path.isfile(plot_file))

    def test_plot_pareto(self):
        # Test that the function runs without errors
        plot_pareto(self.results, self.output_folder)

        # Check if the plot files exist
        plot_file_1 = os.path.join(self.output_folder, 'sizing.png')
        plot_file_2 = os.path.join(self.output_folder, 'correl.png')
        self.assertTrue(os.path.isfile(plot_file_1))
        self.assertTrue(os.path.isfile(plot_file_2))

    def test_evaluate_results(self):
        # Test that the function runs without errors
        evaluate_results("results_test.csv", output_folder=self.output_folder)

        # Check if the plot files exist
        plot_files = [
            os.path.join(self.output_folder, 'solutions.png'),
            os.path.join(self.output_folder, 'pairplot_kpi.png'),
            os.path.join(self.output_folder, 'pairplot_techs.png'),
            os.path.join(self.output_folder, 'sizing.png'),
            os.path.join(self.output_folder, 'correl.png')
        ]
        for plot_file in plot_files:
            self.assertTrue(os.path.isfile(plot_file))


if __name__ == '__main__':
    unittest.main()
