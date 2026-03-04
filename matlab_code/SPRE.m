%% Sparse probabilistic Richardson extrapolation
% Conditions the specified Gaussian process (GP) model on the dataset.
%
% Inputs:
% A    = m x d, binary matrix representing the sparse basis 
% X    = n_train x d, training inputs
% Y    = n_train x 1, training outputs
% x    = p x 1, kernel parameters
% str  = kernel specification (c.f. function "kernel")
% grad = set to true if gradient output is required          %%%%%%%%%%%%% NEW
%
% Outputs:
%
% out.mu      = scalar, predictive mean for f(0)
% out.var     = scalar, predictive variance for f(0)
% out.mu_GP   = function R^d -> R, predictive mean for fitted GP
% out.cov_GP  = function R^d x R^d -> R, predictive covariance for fitted GP
% out.mu_cv   = n_train x 1, LOOCV predictive means
% out.var_cv  = n_train x 1, LOOCV predictive variances
% out.cv      = scalar, LOOCV criterion
% out.cv_grad = p x 1, gradient of LOOCV criterion

function out = SPRE(A,X,Y,x,str,grad)

addpath(genpath("helper_functions"))

%% Initialise

[m,d] = size(A);
n_train = size(X,1);
p = size(x,1);

%% Data normalisation

ep = 10^(-16); % smallest permitted normalising constant
nX = ep + range(X,1); % normalising constant for X
nY = ep + range(Y);% normalising constant for Y
Xn = X ./ nX;
Yn = Y / nY;

%% Calculations

% basis functions
% A = m x d
% X = n x d
% Xs = n_test x d
V = @(A,X) x2fx(X,A);
v = @(A,Xs) x2fx(Xs,A)';

if ~unisolvent(A,X) % i.e. if the basis functions are not linearly independent 
    error('The set X is not unisolvent')
end

% kernel
k = kernel(str,d);

% residual term
% A = m x d
% X = n_train x d
% Xs = n_test x d
% x = p x 1
r = @(A,X,Xs,x) v(A,Xs) - V(A,X)' * inv(k(X,X,x)) * k(X,Xs,x);

% coefficient estimator
% A = m x d
% X = n_train x d
% Y = n_train x 1
% x = p x 1
beta = @(A,X,Y,x) inv( V(A,X)' * inv(k(X,X,x)) * V(A,X) ) * (V(A,X)' * inv(k(X,X,x)) * Y);

% predictive mean
% A = m x d
% X = n_train x d
% Y = n_train x 1
% Xs = n_test x d
% x = p x 1
mu_GP = @(A,X,Y,Xs,x) k(Xs,X,x) * inv(k(X,X,x)) * Y + r(A,X,Xs,x)' * beta(A,X,Y,x);  

% predictive covariance
% A = m x d
% X = n_train x d
% Xs = n_test x d
% x = p x 1
cov_GP = @(A,X,Xs,x) k(Xs,Xs,x) - k(Xs,X,x) * inv(k(X,X,x)) * k(X,Xs,x) ...
                     + r(A,X,Xs,x)' * inv( V(A,X)' * inv(k(X,X,x)) * V(A,X) ) * r(A,X,Xs,x);

% cross-validation local loss (log-likelihood of test data)
% A = m x d
% X = n_train x d
% Y = n_train x 1
% Xs = n_test x d
% Ys = n_test x 1
% x = p x 1
cv_local_loss = @(A,X,Y,Xs,Ys,x) - (1/2) * log(det(2*pi*cov_GP(A,X,Xs,x))) ...
                                 - (1/2) * (Ys - mu_GP(A,X,Y,Xs,x))' * inv(cov_GP(A,X,Xs,x)) * (Ys - mu_GP(A,X,Y,Xs,x));

% LOOCV loss for ith datum
% A = m x d
% X = n_train x d
% Y = n_train x 1
% x = p x 1
% i = index in {1,...,n_train}
cv_loss = @(A,X,Y,x,i) cv_local_loss(A,remove_row(X,i),remove_row(Y,i),X(i,:),Y(i),x);

% LOOCV loss 
% A = m x d
% X = n_train x d
% Y = n_train x 1
% x = p x 1
cv_loss = @(A,X,Y,x) sum( arrayfun( @(i) cv_loss(A,X,Y,x,i), 1:n_train) );

%% Output on original scale

% predictive mean for f(0)
out.mu = nY * mu_GP(A,Xn,Yn,zeros(1,d),x);

% predictive variance for f(0)
out.var = nY^2 * cov_GP(A,Xn,zeros(1,d),x);

% predictive mean function
out.mu_GP = @(Xs) nY * mu_GP(A,Xn,Yn,Xs./nX,x);

% predictive covariance function
out.cov_GP = @(Xs) nY^2 * cov_GP(A,Xn,Xs./nX,x);

% LOOCV predictive mean
out.mu_cv = arrayfun( @(i) nY * mu_GP(A,remove_row(Xn,i),remove_row(Yn,i),Xn(i,:),x), 1:n_train);

% LOOCV predictive variance
out.var_cv = arrayfun( @(i) nY^2 * cov_GP(A,remove_row(Xn,i),Xn(i,:),x), 1:n_train);

% LOOCV criterion and its gradient
out.cv = cv_loss(A,Xn,Yn,x);
if grad                          %%%%%%%%%%%%% NEW
    out.cv_grad = AutoDiffJacobianAutoDiff( @(x) cv_loss(A,Xn,Yn,x) , x);
end

end

