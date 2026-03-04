%% Multivariate Richardson extrapolation
% Based on fitting a multivariate polynomial to the dataset.
% Selects the m data closest to 0 and fits a degree m polynomial specified
% by the matrix A.
%
% Inputs:
% A   = m x d, binary matrix representing the sparse basis 
% X   = n_train x d, training inputs
% Y   = n_train x 1, training outputs
%
% Outputs:
%
% out.mu = scalar, point estimate for f(0)

function out = MRE(A,X,Y)

%% Initialise

[m,d] = size(A);
idx = knnsearch(X,zeros(1,d),'K',m); % find m data closest to 0
X = X(idx,:);
Y = Y(idx);

%% Data normalisation

ep = 10^(-16); % smallest permitted normalising constant
nX = ep + range(X,1); % normalising constant for X
nY = ep + range(Y);% normalising constant for Y
Xn = X ./ nX;
Yn = Y / nY;

%% Polynomial fit
 
V = x2fx(Xn,A);
v = x2fx(zeros(1,d),A);
out.mu = nY * v * ((V' * V) \ (V' * Yn));

end