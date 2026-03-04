%% Extrapolation
% This code extrapolates the value f(0) of a function f : R^d -> R based 
% on training data y_i = f(x_i).
%
% Inputs:
% X       = n_train x d, training (vector) inputs
% Y       = n_train x 1, training (scalar) outputs
% options = options for extrapolation method 
%
% Options:
% options.name      = string, "MRE" (multivariate Richardson extrapolation)
%                             "GRE" (Gauss-Richardson extrapolation)
%                          or "SPRE" (default; sparse probabilistic Richardson extrapolation)
% options.k_name    = string, "Gaussian"
%                             "GaussianARD"
%                             "Matern1/2"
%                             "Matern3/2" 
%                          or "white" (default)
% options.max_order = integer, maximum polynomial order
% options.plot      = logical, plot LOOCV fit if probabilistic method used (default)
%
% Outputs:
% out.mu  = predictive mean for f(0)
% out.var = predictive variance for f(0), if probabilistic method used
% (depending on the extrapolation method, other outputs may be present)
%
% Examples:
% d = 2; % data dimension
% n_train = 5; % number of training data
% X = rand(n_train,d); % training inputs
% Y = 3 + sin(X(:,1)) + sin(X(:,2)); % training outputs
% 
% out = extrapolation(X,Y,[]); % extrapolate with default options
% disp("predict f(0) = " + out.mu + "+/-" + sqrt(out.var)) 
%
% options.name = "SPRE"; % use SPRE
% options.k_name = "Gaussian"; % use Gaussian kernel
% options.plot = true; % plot LOOCV fit
% out = extrapolation(X,Y,options); % extrapolate with custom options
% disp("predict f(0) = " + out.mu + "+/-" + sqrt(out.var)) 

function out = extrapolation(X,Y,options)

% default options, if not provided
if ~isfield(options,"name")
    options.name = "SPRE";
end
if ~isfield(options,"k_name")
    options.k_name = "white";
end
if ~isfield(options,"plot")
    options.plot = true;
end

% call the requested extrapolation method
if strcmp(options.name,"MRE")
    error("not coded yet")
elseif strcmp(options.name,"GRE")
    out = GRE_stepwise(X,Y,options.k_name,options.max_order);  %%%%%%%%% NEW
elseif strcmp(options.name,"SPRE")
    out = SPRE_stepwise(X,Y,options.k_name,options.max_order);  %%%%%%%%% NEW
end

% plot LOOCV fit
n_train = size(X,1);
if options.plot && ~strcmp(options.name,"MRE")
    figure()
    errorbar(1:n_train,out.mu_cv,out.var_cv.^(1/2),'bo'); hold on
    scatter(1:n_train,Y,'kx')
    xticks(1:n_train)
    xlabel('$i$','Interpreter','latex')
    ylabel('$f(\mathbf{x}_i)$','Interpreter','latex')
    legend({'predicted','actual'})
    title("Leave-one-out cross validation (" + options.name + ")")
end

end



