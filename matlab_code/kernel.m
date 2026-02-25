%% Kernel constructor
%
% Inputs:
% str = string, kernel specification
% d   = data dimension
%
% Outputs:
% k  = function R^d x R^d -> R, function handle to the kernel
% x0 = p x 1, default parameter values for the kernel

function [k,x0] = kernel(str,d)

ep = 10^(-16); % smallest allowed kernel amplitude for data that are normalised

if strcmp(str,"Gaussian")

    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) * exp( - pdist2(X1,X2).^2 / (softplus(x(2)).^2) );

    % default hyper-parameter values
    x0 = [1;0.1];

elseif strcmp(str,"GaussianARD")

    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) ...
                   * exp( - cellsum( arrayfun(@(i) pdist2(X1(:,i),X2(:,i)).^2 ...
                                                   / (softplus(x(1+i))^2),1:d, ...
                                              'UniformOutput',false) ) );

    % default hyper-parameter values
    x0 = [1;0.1*ones(d,1)];    

elseif strcmp(str,"white")

    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) * white(X1,X2);

    % default hyper-parameter values
    x0 = [1];

elseif strcmp(str,"Matern1/2")

    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) * exp( - pdist2(X1,X2) / softplus(x(2)) );

    % default hyper-parameter values
    x0 = [1;1];

elseif strcmp(str,"Matern3/2")

    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) * (1 + sqrt(3) * pdist2(X1,X2) / softplus(x(2)) ) ...
                   .* exp( - sqrt(3) * pdist2(X1,X2) / softplus(x(2)) );

    % default hyper-parameter values
    x0 = [1;1];    

elseif iscell(str) % compatability layer for GRE

    % basis functions
    % B = m x d
    B = str{1};

    % convergence rate ansatz b(x)
    b = @(X) sum(x2fx(X,B),2);
    
    % base kernel name
    k_name = str{2};
    [k_base,x0_base] = kernel(k_name,d);
    p_base = length(x0_base);
    
    % X1 = n1 x d
    % X2 = n2 x d
    % x = p x 1
    k = @(X1,X2,x) (ep + softplus(x(1))) * ...
                   b(X1) .* k_base(X1,X2,x(2:(p_base+1))) .* b(X2)';
    
    % default hyper-parameter values
    x0 = [1;x0_base];
       
end

end



