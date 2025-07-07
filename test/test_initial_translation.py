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
       
        a1 = helper_functions.softplus(0)
        a2 = 0.6931472 # = log(1 + exp(0))
        b1 = helper_functions.softplus(1)
        b2 = 1.313262 # = log(1 + exp(1))

        self.assertTrue(round(a1, 6) == round(a2, 6), f"Failed softplus(0) == {a2} (to 6 d.p.)! Result is {a1}")
        self.assertTrue(round(b1, 6) == round(b2, 6), f"Failed softplus(1) == {b2} (to 6 d.p.)! Result is {b1}")

   
   