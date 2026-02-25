%% Stepwise model selection for SPRE
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
% out.x       = p x 1, fitted kernel parameters         %%%%%%%%%%%%% NEW
% out.A       = learned index set                %%%%%%%%%%%%% NEW

function out = SPRE_stepwise(X,Y,k_name,max_order)   %%%%%%%%%%%%% NEW

addpath(genpath("helper_functions"))

[n_train,d] = size(X); % data dimension   

A = zeros(1,d); % initialise with just an intercept
order = 0; 
fit = SPRE_opt(A,X,Y,k_name);
cv = fit.cv;

carry_on = true;
while carry_on && (order < max_order)  %%%%%%%%%%%%% NEW
    m = size(A,1); 
    order = order + 1; % consider the addition of higher-order interations
    A_extra = stepwise(A,order); % all predictors of the next order to consider
    n_extra = size(A_extra,1);

    % monitoring
    cwbar("Fitting interactions of order " + num2str(order,'%u') + ": ")

    to_include = false(n_extra,1);
    parfor i = 1:n_extra
        A_new = [A; A_extra(i,:)];
        if unisolvent(A_new,X)
            fit_new = SPRE_opt(A_new,X,Y,k_name);
            cv_new = fit_new.cv;
            if cv_new < cv % if adding new predictor helped
                to_include(i) = true;
            end
        else
            warning('A non-unisolvent set encountered')
        end
        cwbar(i/n_extra)
    end
    if (sum(to_include) > 0) && ((m + sum(to_include)) < (n_train - 1)) 
        A_updated = [A; A_extra(to_include,:)];
        fit_updated = SPRE_opt(A_updated,X,Y,k_name);
        cv_updated = fit_updated.cv; % calculate cv loss for updated model
        if cv_updated >= cv
            carry_on = false; % terminate
        else 
            fit = fit_updated;
            A = A_updated;
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
grad = false; % don't compute gradient    %%%%%%%%%%%%% NEW
out = SPRE(A,X,Y,x_opt,k_name,grad);  %%%%%%%%%%%%% NEW
out.x = x_opt;        %%%%%%%%%%%%% NEW
out.A = A;            %%%%%%%%%%%%% NEW

end