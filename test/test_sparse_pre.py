###############################################################################
# Tests for Sparse Probabilistic Richardson Extrapolation Python
#
# Richard Howey, July 2025
###############################################################################

# Python modules
import unittest
import jax.numpy as jnp

# Application modules
from sparse_pre import helper_functions
from sparse_pre.MRE import MRE
from sparse_pre.SPRE import SPRE


class SPRETestCase(unittest.TestCase):
    '''
    Test the initial translation from MatLab
    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cellsum(self):
        # Create test data
        arrays = jnp.array([ [0, 1, 2],
                   [1, 2, 3],
                   [5, 6, 7]
        ])

        # Test answer
        ans = jnp.array([6, 9, 12])

        result = helper_functions.cellsum(arrays)
        
        # Check all elements match
        self.assertTrue((result == ans).all(), f"Failed cellsum! Result is {result}")

    def test_remove_row(self):
        # Create test data
        arr = jnp.array([[1, 2, 3], [4, 5, 6], [14, 15, 16]])

        # Test answer
        ans = jnp.array([[1, 2, 3], [14, 15, 16]])

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
        
        A = jnp.array([[0, 0], [0, 1], [1, 0]])
        order = 2 
        ans = jnp.array([[0, 2], [1, 1], [2, 0]])
        result = helper_functions.stepwise(A, order)
        self.assertTrue((result == ans).all(), f"Failed stepwise! Result is {result} not {ans}")

        order = 3
        ans = jnp.array([])
        result = helper_functions.stepwise(A, order)
        self.assertTrue(result.shape[0] == 0, f"Failed stepwise! Result is {result} not {ans}")

        order = 1
        ans = jnp.array([[0, 1], [1, 0]])
        result = helper_functions.stepwise(A, order)
        self.assertTrue((result == ans).all(), f"Failed stepwise! Result is {result} not {ans}")
   
    def test_white(self):

        A = jnp.array([[1, 1], [0, 1], [1, 0]])
        B = jnp.array([[1, 2], [1, 1], [7, 8], [1, 0]])
        ans = jnp.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        result = helper_functions.white(A, B)
        self.assertTrue((result == ans).all(), f"Failed white! Result is {result} not {ans}")

    def test_x2fx(self):

        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        ans = jnp.array([
            [1.0000,    0.0975,    0.0794,    0.6637],
            [1.0000,    0.2785,    0.2523,    0.8205],
            [1.0000,    0.5469,    0.0695,    0.0161],
            [1.0000,    0.9575,    0.8746,    0.8343],
            [1.0000,    0.9649,    0.6102,    0.3999]])
        
        thres = 0.0001
        result = helper_functions.x2fx(X, A)
        self.assertTrue((abs(result - ans) < thres).all(), f"Failed x2fx! Result is {result} not {ans}")

        #A = jnp.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        #ans =   
        #result = helper_functions.x2fx(X, A)
        #self.assertTrue((abs(result - ans) < thres).all(), f"Failed x2fx! Result is {result} not {ans}")


    def test_MRE(self):

        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([[3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]])

        ans = 3.6664
        
        result = MRE(A, X, Y)
        thres = 0.0001

        self.assertTrue((abs(result - ans) < thres).all(), f"Failed MSE! Result is {result} not {ans}")

        A = jnp.array([[0, 0]])
        ans = 3.6467
        result = MRE(A, X, Y)
      
        self.assertTrue((abs(result - ans) < thres).all(), f"Failed MSE! Result is {result} not {ans}")


    def test_kernels(self):

        X1 = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])
        
        X2 = jnp.array([[0.1,     0.2],
                        [0.3,    0.4],
                        [0.5,    0.6]])
          
        # Test Gaussian
        spre = SPRE("Gaussian", X1.shape[1])
        result = spre.kernel(X1, X2)

        ans0 = jnp.array([[0.5126,    0.6903,    0.6964],
                    [0.4024,    0.6594,    0.8096],
                    [1.0555,    1.1967,    1.0165],
                    [0.1413,    0.3801,    0.7660],
                    [0.2739,    0.6049,    1.0006]])
        
        thres = 0.0001
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel Gaussian, default! Result is {result} not {ans0}")

        # Now test with different hyper parameters
        ans1 = jnp.array([
                        [0.5623,    0.6690,    0.6725],
                        [0.4882,    0.6514,    0.7344],
                        [0.8574,    0.9226,    0.8387],
                        [0.2649,    0.4722,    0.7110],
                        [0.3900,    0.6194,    0.8310]])
        
        result = spre.kernel(X1, X2, [0.5, 0.5])

        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel Gaussian, hyperparameters = [0.5, 0.5]! Result is {result} not {ans1}")

        # Test GaussianARD
        spre = SPRE("GaussianARD", X1.shape[1])

        result = spre.kernel(X1, X2)
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel GaussianARD, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, [0.5, 0.5, 0.5])
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel GaussianARD, hyperparameters = [0.5, 0.5, 0.5]! Result is {result} not {ans1}")

        # Test Matern1/2 kernel
        spre = SPRE("Matern1/2", X1.shape[1])
      
        ans0 = jnp.array([
                [0.7578,    0.8335,    0.8361],
                [0.7089,    0.8204,    0.8854],
                [1.0076,    1.1048,    0.9857],
                [0.5634,    0.6986,    0.8662],
                [0.6459,    0.7973,    0.9772]])

        ans1 = jnp.array([
                [0.4642,    0.5277,    0.5300],
                [0.4243,    0.5166,    0.5725],
                [0.6815,    0.7716,    0.6616],
                [0.3112,    0.4159,    0.5558],
                [0.3742,    0.4970,    0.6539]])

        result = spre.kernel(X1, X2)      
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel Matern1/2, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, [0.5, 0.5])       
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel Matern1/2, hyperparameters = [0.5, 0.5]! Result is {result} not {ans1}")

        # Test Matern3/2 kernel
        spre = SPRE("Matern3/2", X1.shape[1])
      
        ans0 = jnp.array([
                [0.9893,    1.0681,    1.0706],
                [0.9335,    1.0551,    1.1165],
                [1.2108,    1.2650,    1.1960],
                [0.7476,    0.9213,    1.0991],
                [0.8564,    1.0315,    1.1900]])

        ans1 = jnp.array([
                [0.6162,    0.6946,    0.6972],
                [0.5632,    0.6814,    0.7451],
                [0.8493,    0.9132,    0.8324],
                [0.4017,    0.5519,    0.7268],
                [0.4935,    0.6576,    0.8256]])

        result = spre.kernel(X1, X2)      
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel Matern3/2, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, [0.5, 0.5]) 
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel Matern3/2, hyperparameters = [0.5, 0.5]! Result is {result} not {ans1}")

        # Test white kernel
        spre = SPRE("white", X1.shape[1])
      
        X1 = jnp.array([[0.1,     0.2],
                        [0.3,    0.4],
                        [0.5,    0.6],
                        [0.1,    0.4],
                        [0.3,    0.4]])
        
        ans0 = jnp.array([
                [1.3133,         0 ,        0],
                [0,    1.3133,         0],
                [0,         0,    1.3133],
                [0,         0,         0],
                [0,    1.3133,         0]])

        ans1 = jnp.array([
                [0.9741,         0 ,        0],
                [0,    0.9741,         0],
                [0,         0,    0.9741],
                [0,         0,         0],
                [0,    0.9741,         0]])
        

        result = spre.kernel(X1, X2)       
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel white, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, jnp.array([0.5]))       
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel white, hyperparameters = [0.5]! Result is {result} not {ans1}")
    
        # Test composite kernel
        B = jnp.zeros((1, 2))        
        spre = SPRE("Gaussian", X1.shape[1], B)

        X1 = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])
        
        X2 = jnp.array([[0.1,     0.2],
                        [0.3,    0.4],
                        [0.5,    0.6]])
        
        ans0 = jnp.array([
                [0.6732,    0.9065,    0.9145],
                [0.5284,    0.8660,    1.0632],
                [1.3862,    1.5716,    1.3349],
                [0.1855,    0.4991,    1.0060],
                [0.3598,    0.7943,    1.3140]])

        ans1 = jnp.array([
                [0.5477,    0.6517,    0.6551],
                [0.4755,    0.6345,    0.7153],
                [0.8352,    0.8987,    0.8170],
                [0.2581,    0.4599,    0.6926],
                [0.3799,    0.6033,    0.8095]])

        result = spre.kernel(X1, X2)      
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel composite_kernel, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, jnp.array([0.5, 0.5, 0.5]))
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel composite_kernel, hyperparameters = [0.5, 0.5, 0.5]! Result is\n {result} not\n {ans1}")
    
    def test_SPRE(self):
  
        A = jnp.array([[0, 0], [0, 1], [1, 1], [1, 0]])    
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 

        x = [0.5, 0.5]

        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_sparse_basis(A)
        spre.set_normalised_data(X, Y)

        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        print(result)

        ans = { "mu": 2.9720,
                "var": 1.8634,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.8350, 4.0527, 3.6050, 32.5128, 4.4413]),
                "var_cv": jnp.array([0.1134, 0.0910, 1.9240, 8.6131e+05, 0.8833]),
                "cv": -9.5990,
                "cv_grad": jnp.array([-1.5961, 4.8491])
            }
      
        thres = 0.0001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed SPRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed SPRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed SPRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed SPRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE, cv! Result is {result['cv']} not {ans['cv']}")
        thres = 0.0001
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")


        # Test 2
        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        spre.set_sparse_basis(A)
       
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        
        ans = { "mu": 3.1208,
                "var": 1.7652,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.5501, 4.5639, 3.8842, 4.8592, 4.2607]),
                "var_cv": jnp.array([0.8758, 2.9239, 0.6543, 0.7242, 0.2691]),
                "cv": -4.4413,
                "cv_grad": jnp.array([-1.4598, 3.7235])
            }
      
        thres = 0.0001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed SPRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed SPRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed SPRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed SPRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE, cv! Result is {result['cv']} not {ans['cv']}")
        thres = 0.1
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")

        # Test 3
        A = jnp.array([[0, 0]])
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        ans = { "mu": 3.5814,
                "var": 0.4423,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.9238, 3.9955, 3.9893, 4.5278, 4.3830]),
                "var_cv": jnp.array([0.0796, 0.0632, 0.6165, 0.1521, 0.1413]),
                "cv": -0.1932,
                "cv_grad": jnp.array([-1.4592, 4.4570])
            }
        
        thres = 0.0001
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE (test 3), cv! Result is {round(result['cv'], 3)} not {ans['cv']}")
        #thres = 0.000001
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE (test 3), cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")

    def test_SPRE_try2(self):
  
        A = jnp.array([[0, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 

        #x = [0.5, 0.5]
        x = [3.245772123336792, 5.657604694366455]

        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_normalised_data(X, Y)
        # Test 3
        
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        ans = { "mu": 3.5814,
                "var": 0.4423,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.9238, 3.9955, 3.9893, 4.5278, 4.3830]),
                "var_cv": jnp.array([0.0796, 0.0632, 0.6165, 0.1521, 0.1413]),
                "cv": -0.1932,
                "cv_grad": jnp.array([-1.4592, 4.4570])
            }

        ''' 
        ans2 = 
                mu: 3.0106
        var: 4.2912e-04
      mu_GP: @(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)
     cov_GP: @(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)
      mu_cv: [3.8474 4.0421 3.6963 4.6408 4.3869]
     var_cv: [4.3486e-04 2.9516e-04 0.0051 7.9681e-04 7.5929e-04]
         cv: 10.3989
    cv_grad: [0.0148 0.0464]

        {'mu': Array([3.0148942], dtype=float32),
        'var': Array([[0.]], dtype=float32),
        'mu_cv': Array([3.8465989, 4.040699 , 3.6942513, 4.641085 , 4.386685 ],
                          dtype=float32),
        'var_cv': Array([3.1281338e-04, 1.0977647e-03, 8.2917316e-03, 3.9802133e-05, 0.0000000e+00],
                 dtype=float32),
        'cv': Array(nan, dtype=float32),
        'cv_grad': Array([ 39.174683, 177.97368 ], dtype=float32)}
    
        print(result)
        '''
        thres = 0.0001
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE (test 3), cv! Result is {round(result['cv'], 3)} not {ans['cv']}")
        #thres = 0.000001
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE (test 3), cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")


    def test_SPRE_opt(self):
        
        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 
        
        ans = {"x": jnp.array([-1.1780, 1.2391]),
              "cv": 0.5987
              }
        
        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_sparse_basis(A)
        spre.set_normalised_data(X, Y)

        result = spre.perform_extrapolation_optimization()

        print(result)

        #self.assertTrue((jnp.round(result['cv'], 1) == jnp.round(ans['cv'], 1)).all(), f"Failed SPRE_opt, cv! Result is {result['cv']} not {ans['cv']}")
        #self.assertTrue((jnp.round(result['x'], 2) == jnp.round(ans['x'], 2)).all(), f"Failed SPRE_opt, x! Result is {result['x']} not {ans['x']}")
        to_show = ['x', 'cv']
        for field in to_show:
            print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

        # Test 2  
        A = jnp.array([[0, 0]])
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation_optimization()
        print(result)
        ans = {"x": jnp.array([1.3712, 4.6923]),
                "cv": -10.2897
              }
        
        #self.assertTrue((jnp.round(result['cv'], 1) == jnp.round(ans['cv'], 1)).all(), f"Failed SPRE_opt (test 2), cv! Result is {result['cv']} not {ans['cv']}")

        to_show = ['x', 'cv']
        for field in to_show:
            print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

    def test_SPRE_opt_try(self):
        '''
        Test to output loss, gradient and hyper-parameters to help develop the code to fit the hyperparameters
        '''
        
        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 
        
        ans = {"x": jnp.array([-1.1780, 1.2391]),
              "cv": 0.5987
              }
        
        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_sparse_basis(A)
        spre.set_normalised_data(X, Y)

        result = spre.perform_extrapolation_optimization()

        #print(result)
        #to_show = ['x', 'cv']
        #for field in to_show:
        #    print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

    def test_SPRE_opt_try2(self):
        '''
        Test to output loss, gradient and hyper-parameters to help develop the code to fit the hyperparameters
        '''
        
        A = jnp.array([[0, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 
        
        ans = {"x": jnp.array([-1.1780, 1.2391]),
              "cv": 0.5987
              }
        
        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_sparse_basis(A)
        spre.set_normalised_data(X, Y)

        result = spre.perform_extrapolation_optimization()


    def test_SPRE_stepwise(self):

        #A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 
    
        # Set up SPRE object
        spre = SPRE("Gaussian", X.shape[1])

        # Set data
        spre.set_normalised_data(X, Y)

        result = spre.stepwise_selection()

        ans = { "mu": 2.9892,
                "var": 7.6288e-04,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.8446, 4.0436, 3.6829, 4.6440, 4.3873]),
                "var_cv": jnp.array([4.0282e-04, 2.7214e-04, 0.0047, 7.3606e-04, 6.9963e-04]),
                "cv": 10.6180,
                "cv_grad": jnp.array([0.0169, -0.0320])
            }
        
        print("ans = ")
        print(ans)

        print("result = ")
        print(result)

    def test_GRE_stepwise(self):

        #A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 
    
        # Set up SPRE object
        spre = SPRE("Gaussian", X.shape[1], jnp.zeros((1, X.shape[1]), dtype=int))

        # Set data
        spre.set_normalised_data(X, Y)

        result = spre.stepwise_selection()
 
        ans = { "mu": 3.0208,
                "var": 5.0898e-04,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.8483, 4.0416, 3.7011, 4.6396, 4.3868]),
                "var_cv": jnp.array([4.3453e-04, 2.9556e-04, 0.0050, 7.9684e-04, 7.5886e-04]),
                "cv": 10.3088,
                "cv_grad": jnp.array([0.0783, 0.0804, 0.0281])
            }
        
        print(result)
        to_show = ['mu', 'var', 'mu_cv', 'var_cv', 'cv', 'cv_grad']
        for field in to_show:
            print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

    def test_GRE(self):

        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        X = jnp.array([[0.8147,    0.0975],
                        [0.9058,    0.2785],
                        [0.1270,    0.5469],
                        [0.9134,    0.9575],
                        [0.6324,    0.9649]])

        Y = jnp.array([3.8249,
                        4.0618,
                        3.6467,
                        4.6093,
                        4.4130]) 

        x = [0.9, 0.5, 0.5]
        B = jnp.array([[0.55, 0.66]])
       
        # Set up SPRE object
        spre = SPRE("Gaussian", X.shape[1], B)

        # Set data
        spre.set_normalised_data(X, Y)
        spre.set_sparse_basis(A)

        result = spre.perform_extrapolation(x, return_mu_and_var=True)

        ans = { "mu": 3.2575,
                "var": 2.1722,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.5501, 4.5639, 3.8842, 4.8592, 4.2607]),
                "var_cv": jnp.array([0.5933, 1.9808, 0.4433, 0.4906, 0.1823]),
                "cv": -3.5704,
                "cv_grad": jnp.array([-1.2498, -1.3942, 3.2989])
            }
      
        thres = 0.001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed SPRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed SPRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed SPRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed SPRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE, cv! Result is {result['cv']} not {ans['cv']}")
        thres = 0.01
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']}")
