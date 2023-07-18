import unittest
import sys
import os
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.append(path)
from src.financial import calculate_annualized_npv

class TestCalculateAnnualizedNPV(unittest.TestCase):
    def test_annualized_npv(self):
        capital_cost = 100000
        lifetime = 5
        repair_cost = 20000
        repair_interval = 1
        discount_rate = 0.1

        expected_output = -46379.74
        result = calculate_annualized_npv(capital_cost, lifetime, repair_cost, repair_interval, discount_rate)

        # Assert that the result is approximately equal to the expected output
        self.assertAlmostEqual(result, expected_output, places=1)

        # Add more test cases as needed

if __name__ == '__main__':
    unittest.main()