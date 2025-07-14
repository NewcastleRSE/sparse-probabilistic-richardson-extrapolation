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
from src.initial_translation import SPRE
from src.initial_translation.kernel import kernel

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
        
        result = helper_functions.x2fx(X, A)
        round_result = np.round(result, 4)
        round_ans = np.round(ans, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed x2fx! Result is {round_result} not {round_ans}")

    def test_MRE(self):

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

        ans = 3.6664
        
        result = MRE.MRE(A, X, Y)
        round_result = np.round(result, 4)
        round_ans = np.round(ans, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed MSE! Result is {round_result} not {round_ans}")

    def test_kernels(self):

        X1 = np.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])
        
        X2 = np.array([[0.1,     0.2],
                        [0.3,    0.4],
                        [0.5,    0.6]])
        
        # Test Gaussian
        gauss = kernel("Gaussian", 2)

        result0 = gauss(X1, X2)

        ans0 = np.array([[0.5126,    0.6903,    0.6964],
                    [0.4024,    0.6594,    0.8096],
                    [1.0555,    1.1967,    1.0165],
                    [0.1413,    0.3801,    0.7660],
                    [0.2739,    0.6049,    1.0006]])
        
        round_result = np.round(result0, 4)
        round_ans = np.round(ans0, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Gaussian, default! Result is {round_result} not {round_ans}")

        # Now test with different hyper parameters
        ans1 = np.array([
                        [0.5623,    0.6690,    0.6725],
                        [0.4882,    0.6514,    0.7344],
                        [0.8574,    0.9226,    0.8387],
                        [0.2649,    0.4722,    0.7110],
                        [0.3900,    0.6194,    0.8310]])
        
        result1 = gauss(X1, X2, [0.5, 0.5])

        round_result = np.round(result1, 4)
        round_ans = np.round(ans1, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Gaussian, hyperparameters = [0.5, 0.5]! Result is {round_result} not {round_ans}")

        # Test GaussianARD
        gauss_ard = kernel("GaussianARD", 2)

        result0 = gauss_ard(X1, X2)
        round_result = np.round(result0, 4)
        round_ans = np.round(ans0, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel GaussianARD, default! Result is {round_result} not {round_ans}")

        result1 = gauss_ard(X1, X2, [0.5, 0.5, 0.5])
        round_result = np.round(result1, 4)
        round_ans = np.round(ans1, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel GaussianARD, hyperparameters = [0.5, 0.5, 0.5]! Result is {round_result} not {round_ans}")

        # Test Matern1/2 kernel
        matern1_2 = kernel("Matern1/2", 2)

        ans0 = np.array([
                [0.7578,    0.8335,    0.8361],
                [0.7089,    0.8204,    0.8854],
                [1.0076,    1.1048,    0.9857],
                [0.5634,    0.6986,    0.8662],
                [0.6459,    0.7973,    0.9772]])

        ans1 = np.array([
                [0.4642,    0.5277,    0.5300],
                [0.4243,    0.5166,    0.5725],
                [0.6815,    0.7716,    0.6616],
                [0.3112,    0.4159,    0.5558],
                [0.3742,    0.4970,    0.6539]])

        result0 = matern1_2(X1, X2)
        round_result = np.round(result0, 4)
        round_ans = np.round(ans0, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Matern1/2, default! Result is {round_result} not {round_ans}")

        result1 = matern1_2(X1, X2, [0.5, 0.5])
        round_result = np.round(result1, 4)
        round_ans = np.round(ans1, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Matern1/2, hyperparameters = [0.5, 0.5]! Result is {round_result} not {round_ans}")

        # Test Matern3/2 kernel
        matern3_2 = kernel("Matern3/2", 2)

        ans0 = np.array([
                [0.9893,    1.0681,    1.0706],
                [0.9335,    1.0551,    1.1165],
                [1.2108,    1.2650,    1.1960],
                [0.7476,    0.9213,    1.0991],
                [0.8564,    1.0315,    1.1900]])

        ans1 = np.array([
                [0.6162,    0.6946,    0.6972],
                [0.5632,    0.6814,    0.7451],
                [0.8493,    0.9132,    0.8324],
                [0.4017,    0.5519,    0.7268],
                [0.4935,    0.6576,    0.8256]])

        result0 = matern3_2(X1, X2)
        round_result = np.round(result0, 4)
        round_ans = np.round(ans0, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Matern3/2, default! Result is {round_result} not {round_ans}")

        result1 = matern3_2(X1, X2, [0.5, 0.5])
        round_result = np.round(result1, 4)
        round_ans = np.round(ans1, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel Matern3/2, hyperparameters = [0.5, 0.5]! Result is {round_result} not {round_ans}")

        # Test white kernel
        white_kernel = kernel("white", 2)

        X1 = np.array([[0.1,     0.2],
                        [0.3,    0.4],
                        [0.5,    0.6],
                        [0.1,    0.4],
                        [0.3,    0.4]])
        
        ans0 = np.array([
                [1.3133,         0 ,        0],
                [0,    1.3133,         0],
                [0,         0,    1.3133],
                [0,         0,         0],
                [0,    1.3133,         0]])

        ans1 = np.array([
                [0.9741,         0 ,        0],
                [0,    0.9741,         0],
                [0,         0,    0.9741],
                [0,         0,         0],
                [0,    0.9741,         0]])
        

        result0 = white_kernel(X1, X2)
        round_result = np.round(result0, 4)
        round_ans = np.round(ans0, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel white, default! Result is {round_result} not {round_ans}")

        result1 = white_kernel(X1, X2, [0.5])
        round_result = np.round(result1, 4)
        round_ans = np.round(ans1, 4)
        self.assertTrue((round_result == round_ans).all(), f"Failed kernel white, hyperparameters = [0.5]! Result is {round_result} not {round_ans}")
    
    def test_SPRE(self):

        A = np.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = np.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = np.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 

        x = [0.5, 0.5]
        result = SPRE.SPRE(A, X, Y, x, "Gaussian")
        print(result)

        ans = { "mu": 3.1208,
                "var": 1.7652,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": [3.5501, 4.5639, 3.8842, 4.8592, 4.2607],
                "var_cv": [0.8758, 2.9239, 0.6543, 0.7242, 0.2691],
                "cv": -4.4413,
                "cv_grad": [-1.4598, 3.7235]
            }
      
        self.assertTrue((round(result["mu"][0], 4) == round(ans["mu"], 4)).all(), f"Failed SPRE, mu! Result is {result["mu"][0]} not {ans["mu"]}")
        self.assertTrue((round(result["var"][0][0], 4) == round(ans["var"], 4)).all(), f"Failed SPRE, var! Result is {result["var"][0][0]} not {ans["var"]}")
        self.assertTrue((np.round(result["mu_cv"].flatten(), 4) == np.round(ans["mu_cv"], 4)).all(), f"Failed SPRE, mu_cv! Result is {result["mu_cv"]} not {ans["mu_cv"]}")
        self.assertTrue((np.round(result["var_cv"].flatten(), 4) == np.round(ans["var_cv"], 4)).all(), f"Failed SPRE, var_cv! Result is {result["var_cv"]} not {ans["var_cv"]}")
        self.assertTrue((round(result["cv"], 4) == round(ans["cv"], 4)).all(), f"Failed SPRE, cv! Result is {result["cv"]} not {ans["cv"]}")
        self.assertTrue((round(result["cv_grad"], 4) == round(ans["cv_grad"], 4)).all(), f"Failed SPRE, cv_grad! Result is {result["cv_grad"]} not {ans["cv_grad"]}")
        ##Grad...