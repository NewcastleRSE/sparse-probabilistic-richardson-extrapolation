###############################################################################
# Tests for the initial translation from MatLab to Python
#
# Richard Howey, July 2025
###############################################################################

# Python modules
import unittest
import numpy as np

# Application modules
from src.initial_translation import helper_functions
from src.initial_translation import MRE

class InitialTranslationTestCase(unittest.TestCase):
    '''
    Test the initial translation from MatLab
    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cellsum(self):
        # Create test data
        arrays = [ [0, 1, 2],
                   [1, 2, 3],
                   [5, 6, 7]
        ]

        # Test answer
        ans = [6, 9, 12]

        result = helper_functions.cellsum(arrays)
        
        # Check all elements match
        self.assertTrue((result == ans).all(), f"Failed cellsum! Result is {result}")

    def test_remove_row(self):
        # Create test data
        arr = np.array([[1, 2, 3], [4, 5, 6], [14, 15, 16]])

        # Test answer
        ans = np.array([[1, 2, 3], [14, 15, 16]])

        result = helper_functions.remove_row(arr, 1)
        
        # Check all elements match
        self.assertTrue((result == ans).all(), f"Failed remove_row! Result is {result}")
      
    def test_softplus(self):
        # Check correct answers are returned.
        a1 = helper_functions.softplus(0)
        a2 = 0.6931472 # = log(1 + exp(0))
        b1 = helper_functions.softplus(1)
        b2 = 1.313262 # = log(1 + exp(1))

        self.assertTrue(round(a1, 6) == round(a2, 6), f"Failed softplus(0) == {a2} (to 6 d.p.)! Result is {a1}")
        self.assertTrue(round(b1, 6) == round(b2, 6), f"Failed softplus(1) == {b2} (to 6 d.p.)! Result is {b1}")

    def test_stepwise(self):
        
        A = np.array([[0, 0], [0, 1], [1, 0]])
        order = 2 
        ans = np.array([[0, 2], [1, 1], [2, 0]])
        result = helper_functions.stepwise(A, order)
        self.assertTrue((result == ans).all(), f"Failed stepwise! Result is {result} not {ans}")

        order = 3
        ans = np.array([])
        result = helper_functions.stepwise(A, order)
        self.assertTrue(result.shape[0] == 0, f"Failed stepwise! Result is {result} not {ans}")

        order = 1
        ans = np.array([[0, 1], [1, 0]])
        result = helper_functions.stepwise(A, order)
        self.assertTrue((result == ans).all(), f"Failed stepwise! Result is {result} not {ans}")
   
    def test_white(self):

        A = np.array([[1, 1], [0, 1], [1, 0]])
        B = np.array([[1, 2], [1, 1], [7, 8], [1, 0]])
        ans = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        result = helper_functions.white(A, B)
        self.assertTrue((result == ans).all(), f"Failed white! Result is {result} not {ans}")

    def test_x2fx(self):

        A = np.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = np.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = np.array([[3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]])

        ans = np.array([
            [1.0000,    0.0975,    0.0794,    0.6637],
            [1.0000,    0.2785,    0.2523,    0.8205],
            [1.0000,    0.5469,    0.0695,    0.0161],
            [1.0000,    0.9575,    0.8746,    0.8343],
            [1.0000,    0.9649,    0.6102,    0.3999]])
        
        result = MRE.x2fx(X, A)
        round_result = np.round(result, 4)
        round_ans = np.round(ans, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed x2fx! Result is {round_result} not {round_ans}")