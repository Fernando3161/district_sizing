import unittest
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem

class NSGA2Test(unittest.TestCase):

    def test_nsga2_optimization(self):
        problem = get_problem("sphere")
        algorithm = NSGA2(pop_size=100)
        result = minimize(problem,
                          algorithm,
                          termination=('n_gen', 100),
                          seed=1)
        self.assertAlmostEqual(result.F[0], 0, 4)
        self.assertEqual(len(result.X), 10)  # Check population size
        self.assertEqual(len(result.F), 1)  # Check objective values size

        # Additional assertions or checks can be added based on the problem or expected results

if __name__ == '__main__':
    unittest.main()
