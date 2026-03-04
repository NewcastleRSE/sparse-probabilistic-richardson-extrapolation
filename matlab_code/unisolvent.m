% Check whether a given design set X is A-unisolvent.
%
% Inputs:
% A   = m x d, binary matrix representing the sparse basis 
% X   = n_train x d, training inputs
%
% Outputs:
%
% out = logical (true = unisolvent)

function out = unisolvent(A,X)

% dimensions
[m,~] = size(A);

% basis functions
% A = m x d
% X = n x d
V = @(A,X) x2fx(X,A);

% check unisolvency
out = (rank(V(A,X)) == m);