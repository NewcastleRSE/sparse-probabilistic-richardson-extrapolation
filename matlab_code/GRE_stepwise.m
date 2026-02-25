%% Stepwise model selection for GRE
%
% Inputs:
% X         = n_train x d, training inputs
% Y         = n_train x 1, training outputs
% k_name    = string, "Gaussian"
%                     "GaussianARD"
%                     "Matern1/2"
%                     "Matern3/2" 
%                  or "white" 
% max_order = integer, maximum polynomial order
%
% Outputs:
%
% out.mu      = scalar, predictive mean for f(0)
% out.cov     = scalar, predictive variance for f(0)
% out.mu_GP   = function R^d -> R, predictive mean for fitted GP
% out.cov_GP  = function R^d x R^d -> R, predictive covariance for fitted GP
% out.mu_cv   = n_train x 1, LOOCV predictive means
% out.var_cv  = n_train x 1, LOOCV predictive variances
% out.cv      = scalar, LOOCV criterion
% out.cv_grad = p x 1, gradient of LOOCV criterion

function out = GRE_stepwise(X,Y,k_name,max_order)  %%%%% NEW

addpath(genpath("helper_functions"))

d = size(X,2); % data dimension
 
A = zeros(1,d); % mean function is just the intercept

B = zeros(1,d); % initialise rate function with just an intercept
order = 0; 
fit = SPRE_opt(A,X,Y,{B,k_name});
cv = fit.cv;

carry_on = true;
while carry_on && (order < max_order)   %%%%%%%%%%%%% NEW
    order = order + 1; % consider the addition of higher-order interations
    B_extra = stepwise(B,order); % all predictors of the next order to consider
    n_extra = size(B_extra,1);

    % monitoring
    cwbar("Fitting interactions of order " + num2str(order,'%u') + ": ")

    to_include = false(n_extra,1);
    for i = 1:n_extra
        B_new = [B; B_extra(i,:)];
        fit_new = SPRE_opt(A,X,Y,{B_new,k_name});
        cv_new = fit_new.cv;
        if cv_new < cv % if adding new predictor helped
            to_include(i) = true;
        end
        cwbar(i/n_extra)
    end
    if sum(to_include) > 0
        B_updated = [B; B_extra(to_include,:)];
        fit_updated = SPRE_opt(A,X,Y,{B_updated,k_name});
        cv_updated = fit_updated.cv; % calculate cv loss for updated model
        if cv_updated >= cv
            carry_on = false; % terminate
        else 
            fit = fit_updated;
            B = B_updated;
            cv = cv_updated;
        end
    else
        carry_on = false; % terminate
    end

    cwbar("!")

end

% optimal parameters
x_opt = fit.x;

% fit GP with optimal parameters
out = SPRE(A,X,Y,x_opt,{B,k_name},false);

end