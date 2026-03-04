%% Optimise kernel parameters for SPRE
%
% Inputs:
% A   = m x d, binary matrix representing the sparse basis 
% X   = n_train x d, training inputs
% Y   = n_train x 1, training outputs
% str = kernel specification (c.f. function "kernel")
%
% Outputs:
%
% out.x  = p x 1, fitted kernel parameters 
% out.cv = scalar, LOOCV criterion

function out = SPRE_opt(A,X,Y,str)

addpath(genpath("helper_functions"))

d = size(X,2); % data dimension

% initial parameters
[~,x0] = kernel(str,d); % get default kernel parameters

options = optimoptions('fminunc', ...
                        'Display','off', ... % 'off' or 'iter'
                        'Algorithm','trust-region', ...
                        'SpecifyObjectiveGradient',true, ...
                        'UseParallel',true, ...
                        'MaxIterations',10); %%%%%%%%%%%%%%%% set to 10 for speed
fun = @(x) cv(A,X,Y,x,str);

try
    [out.x,out.cv] = fminunc(fun,x0,options);
catch
    disp(A)
    disp(X)
    disp(Y)
    disp(x0)
    disp(str)
    error("Optimisation failed.")
end

end

% LOOCV (negative log likelihood of held-out datum) and its gradient
function [cv_value,cv_grad] = cv(A,X,Y,x,str)

grad = true; % compute gradient
out = SPRE(A,X,Y,x,str,grad);
cv_value = - out.cv;
cv_grad = - out.cv_grad;

end