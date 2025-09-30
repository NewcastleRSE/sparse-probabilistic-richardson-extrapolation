###############################################################################
# Tests for Sparse Probabilistic Richardson Extrapolation Python
#
# Richard Howey, July 2025
###############################################################################

# Python modules
import unittest
import jax.numpy as jnp

# Ensure 64-bit accuracy is used
from jax import config
config.update("jax_enable_x64", True)

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
            [1.000000000000000,   0.097500000000000,   0.079433250000000,   0.663736090000000],
            [1.000000000000000,   0.278500000000000,   0.252265300000000,   0.820473640000000],
            [1.000000000000000,   0.546900000000000,   0.069456300000000,   0.016129000000000],
            [1.000000000000000,   0.957500000000000,   0.874580500000000,   0.834299560000000],
            [1.000000000000000,   0.964900000000000,   0.610202760000000,   0.399929760000000]])
        
        thres = 0.00000001
        result = helper_functions.x2fx(X, A)
        self.assertTrue((abs(result - ans) < thres).all(), f"Failed x2fx! Result is {result} not {ans}")

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

        ans = 3.666432116526238
        
        result = MRE(A, X, Y)
        thres = 0.00000001

        self.assertTrue((abs(result - ans) < thres).all(), f"Failed MSE! Result is {result} not {ans}")

        A = jnp.array([[0, 0]])
        ans = 3.646700000000000
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

        ans0 = jnp.array(
                    [[0.512604621115517,   0.690250293225782,   0.696355935200161],
                    [0.402368986837116,   0.659403920353982,   0.809616124565160],
                    [1.055514502223468,   1.196689813622174,   1.016481743317268],
                    [0.141283329393321,   0.380072124033268,   0.766022831597360],
                    [0.273936932654505,   0.604856699707559,   1.000587334767931]])

        thres = 0.00000001
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel Gaussian, default! Result is {result} not {ans0}")

        # Now test with different hyper parameters
        ans1 = jnp.array([
                        [0.562320545193706,   0.669040568527791,   0.672490440840407],
                        [0.488169157124197,   0.651413624176236,   0.734359130961065],
                        [0.857389243894399,   0.922607279863801,   0.838727503160834],
                        [0.264921355522316,   0.472183785975813,   0.711001198280384],
                        [0.389991783812596,   0.619379645702218,   0.831043127113665]])
        
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
                [0.757848353071893,   0.833528025006093,   0.836135071840247],
                [0.708944801094794,   0.820397263985140,   0.885403272821872],
                [1.007592202541612,   1.104835012086959,   0.985727448208178],
                [0.563352418107892,   0.698612980396716,   0.866185735522361],
                [0.645888919359929,   0.797256006840289,   0.977195184751785]])

        ans1 = jnp.array([
                [0.464175533378384,   0.527733260264760,   0.529959835417168],
                [0.424252793465585,   0.516555745628809,   0.572487231634953],
                [0.681489779776267,   0.771622408753430,   0.661627669188982],
                [0.311192929727844,   0.415938214298771,   0.555798309505887],
                [0.374182415401730,   0.497008456353754,   0.653918229140656]])

        result = spre.kernel(X1, X2)      
        self.assertTrue((abs(result - ans0) < thres).all(), f"Failed kernel Matern1/2, default! Result is {result} not {ans0}")

        result = spre.kernel(X1, X2, [0.5, 0.5])       
        self.assertTrue((abs(result - ans1) < thres).all(), f"Failed kernel Matern1/2, hyperparameters = [0.5, 0.5]! Result is {result} not {ans1}")

        # Test Matern3/2 kernel
        spre = SPRE("Matern3/2", X1.shape[1])
      
        ans0 = jnp.array([
                [0.989300651962319,   1.068097261315686,   1.070640416598142],
                [0.933515154006332,   1.055111040050575,   1.116462379590273],
                [1.210815690179845,   1.264950236591299,   1.196030230893213],
                [0.747623297276282,   0.921268437125303,   1.099100118478064],
                [0.856408590360751,   1.031514047484945,   1.190007053460509]])

        ans1 = jnp.array([
                [0.616159863991410,   0.694635118926340,   0.697244065760459],
                [0.563246217652787,   0.681389082812582,   0.745112520898125],
                [0.849299642584885,   0.913188602470593,   0.832422303619359],
                [0.401738882561339,   0.551912342887426,   0.726779856997827],
                [0.493540176456098,   0.657640750969894,   0.825609946235218]])

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
                [1.313261687518223,                   0,                   0],
                [0,   1.313261687518223,                   0],
                [0,                   0,   1.313261687518223],
                [0,                   0,                   0],
                [0,   1.313261687518223,                   0]])

        ans1 = jnp.array([
                [0.974076984180107,                   0,                   0],
                [0,   0.974076984180107,                   0],
                [0,                   0,   0.974076984180107],
                [0,                   0,                   0],
                [0,   0.974076984180107,                   0]])
        

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
                [0.673184009755803,   0.906479264891639,   0.914497570574293],
                [0.528415774658709,   0.865969905200202,   1.063237837988406],
                [1.386166756389949,   1.571566884073324,   1.334906529560301],
                [0.185541983577266,   0.499134158986564,   1.005988436501036],
                [0.359750878451421,   0.794335130164652,   1.314033011766694]])

        ans1 = jnp.array([
                [0.547743500804798,   0.651697019285695,   0.655057460503774],
                [0.475514340341283,   0.634527018491422,   0.715322327591678],
                [0.835163128961118,   0.898690516752343,   0.816985156827816],
                [0.258053795032083,   0.459943358222065,   0.692569902969398],
                [0.379882020631194,   0.603323457348160,   0.809499982982484]])

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
        #print(result)

        ans = { "mu": 2.972029043308896,
                "var": 1.863371223464479,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.835023248627188, 4.052727997908001, 3.604995353414705, 32.512808890799015, 4.441257337634016]),
                "var_cv": jnp.array([0.113366186242808, 0.091043724018610, 1.924033461438421, 8.613137354814766e+05, 0.883295894578913]),
                "cv": -9.599036860848194,
                "cv_grad": jnp.array([-1.596112431953308, 4.84909193199531])
            }
      
        # Differences commented out due to matrix instability due to near singular matrix.
        thres = 0.00000001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed SPRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed SPRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        # self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed SPRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        # self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed SPRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE, cv! Result is {result['cv']} not {ans['cv']}")
        thres = 0.0001
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")


        # Test 2
        A = jnp.array([[0, 0], [0, 1], [1, 1], [2, 0]])
        spre.set_sparse_basis(A)
        x = [0.5, 0.5]
    
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        
        ans = { "mu": 3.120814707169809,
                "var": 1.765226199913399,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.550102948166744, 4.563898442086107, 3.884222266902478, 4.859184345320202, 4.260667213941661]),
                "var_cv": jnp.array([0.875813579069014, 2.923918640402877, 0.654329039973795, 0.724211864025656, 0.269137557334076]),
                "cv": -4.441318381789626,
                "cv_grad": jnp.array([-1.459818750418890, 3.723462330567492])
            }
      
        thres = 0.00000001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed SPRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed SPRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed SPRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed SPRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE, cv! Result is {result['cv']} not {ans['cv']}")
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")

        # Test 3
        A = jnp.array([[0, 0]])
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
        ans = { "mu": 3.581399043938413,
                "var": 0.442290292657649,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.923782245412006, 3.995507424574416, 3.989328508085889, 4.527798114682692, 4.382993638244039]),
                "var_cv": jnp.array([0.079598401678480, 0.063155603183687, 0.616516530855601, 0.152072316546668, 0.141272125170928]),
                "cv": -0.193199706494153,
                "cv_grad": jnp.array([-1.459247478540389, 4.456958264941119])
            }
        
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE (test 3), cv! Result is {round(result['cv'], 3)} not {ans['cv']}")
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed SPRE (test 3), cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']} -- differences: {result['cv_grad'] - ans['cv_grad']}")


    def test_SPRE_again(self):
  
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

        x = [3.245772123336792, 5.657604694366455]

        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_normalised_data(X, Y)
        # Test 3
        
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation(x, return_mu_and_var=True)
       
        ans = {"mu": 3.010569826795539,
                "var": 4.291200850578709e-04,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.847364164082203, 4.042131067700186, 3.696305071782295, 4.640838689958968, 4.386904339170414]),
                "var_cv": jnp.array([4.348589368436001e-04, 2.951591731441563e-04, 0.005078675521808, 7.968098945826360e-04, 7.592869926159027e-04]),
                "cv": 10.398882255754817,
                "cv_grad": jnp.array([0.014786163213055, 0.046412966963239])
            }
        
        #print("Python\n")
        #print(result)
        #print("MatLab\n")
        #print(ans)

        # Problem case is a little different due to numerical differences
        thres = 0.000001
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed SPRE (test 3), cv! Result is {round(result['cv'], 3)} not {ans['cv']}")   
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
        
        ans = {"x": jnp.array([0.111128864888310, 2.044895886000445]),
              "cv": 0.598658989966088
              }
        
        # Set up SPRE object
        spre = SPRE("Gaussian", A.shape[1])

        # Set data
        spre.set_sparse_basis(A)
        spre.set_normalised_data(X, Y)

        result = spre.perform_extrapolation_optimization()

        print("Results expected to be slightly different as there are different local minima\n")
        print("Fitting 1\n")
        to_show = ['x', 'cv']
        for field in to_show:
            print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

        # Test 2  
        print("Fitting 2\n")
        A = jnp.array([[0, 0]])
        spre.set_sparse_basis(A)
        result = spre.perform_extrapolation_optimization()
        print(result)
        ans = {"x": jnp.array([52.120665117681909, 11.562586645588107]),
                "cv": -10.556374847056839
              }
        
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

        ans = { "mu": 3.010086212315065,
                "var": 6.034893643589423e-04,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.844504923803925, 4.042882377903229, 3.704198759026446, 4.639408666897037, 4.390116802855905]),
                "var_cv": jnp.array([4.010573807246045e-04, 2.708123608152266e-04, 0.004585667219501, 7.278213345428058e-04, 6.947985730070089e-04]),
                "cv": 10.673544116540279,
                "cv_grad": jnp.array([1.598326953594631e-06, -2.703384092833661e-06])
            }
        
        print("SPRE stepwise fitting expected to be slightly different")
        to_show = ['mu', 'var', 'mu_cv', 'var_cv', 'cv', 'cv_grad']
        for field in to_show:
            print(f"{field}\n Matlab = {ans[field]}\n Python = {result[field]}\n")

    def test_GRE_stepwise(self):

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
 
        ans = { "mu": 2.990043386282295,
                "var": 2.488314869402401e-04,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.845434559505167, 4.043135653510999, 3.686747185266223, 4.643203996005010, 4.387084037710372]),
                "var_cv": jnp.array([4.105994977779579e-04, 2.775380410101257e-04, 0.004844457911238, 7.511885704873892e-04, 7.166650015626653e-04]),
                "cv": 10.550496467554609,
                "cv_grad": jnp.array([0.002494414781713, 0.006961548679049, -0.005055578055596])
            }
        
        print("GRE stepwise fitting expected to be slightly different")
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

        ans = { "mu": 3.257507326688007,
                "var": 2.172222509848281,
                "mu_GP": "@(Xs)nY*mu_GP(A,Xn,Yn,Xs./nX,x)",
                "cov_GP": "@(Xs)nY^2*cov_GP(A,Xn,Xs./nX,x)",
                "mu_cv": jnp.array([3.550102948166775, 4.563898442086371, 3.884222266902433, 4.859184345320324, 4.260667213941578]),
                "var_cv": jnp.array([0.593323923761889, 1.980822085823275, 0.443278207493942, 0.490620646982133, 0.182328471909369]),
                "cv": -3.570413834318558,
                "cv_grad": jnp.array([-1.249775859781986, -1.394237323594431, 3.298938516057062])
            }
      
        thres = 0.00000001
        self.assertTrue((abs(result["mu"][0] - ans["mu"]) < thres).all(), f"Failed GRE, mu! Result is {result['mu'][0]} not {ans['mu']}")
        self.assertTrue((abs(result["var"][0][0] - ans["var"]) < thres).all(), f"Failed GRE, var! Result is {result['var'][0][0]} not {ans['var']}")
        self.assertTrue((abs(result["mu_cv"].flatten() - ans["mu_cv"]) < thres).all(), f"Failed GRE, mu_cv! Result is {result['mu_cv']} not {ans['mu_cv']}")
        result_var_cv = result['var_cv']
        ans_var_cv = ans['var_cv']
        self.assertTrue((abs(result_var_cv - ans_var_cv) < thres).all(), f"Failed GRE, var_cv! Result is {result_var_cv} not {ans_var_cv}")
        self.assertTrue((abs(result['cv'] - ans['cv']) < thres).all(), f"Failed GRE, cv! Result is {result['cv']} not {ans['cv']}")
        thres = 0.01
        self.assertTrue((abs(result['cv_grad'] - ans['cv_grad']) < thres).all(), f"Failed GRE, cv_grad! Result is {result['cv_grad']} not {ans['cv_grad']}")
