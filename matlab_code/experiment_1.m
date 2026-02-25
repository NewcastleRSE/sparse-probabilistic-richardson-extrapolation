%% Visualise 2D extrapolation and dataset

set(groot,'defaultAxesTickLabelInterpreter','latex'); 
set(groot,'defaultLegendInterpreter','latex');
set(gcf,'Color','white')

clear all
rng(1)
addpath(genpath("helper_functions"))

% plotting
RGB = orderedcolors("gem"); % colour palette
HEX = rgb2hex(RGB); % hex colours
figure()
tiledlayout(1,3)

% dimension
d = 2; 

% smoothness
s = 1;

% target
f = @(X) 1000 * (Riemann_Sums(X,s) - 1); % Riemann sums numerical method (improve scalaing for visualisation)
f0 = f(zeros(1,d)); % true value of the integral

% sparse basis, for SPRE
A = table2array(combinations(0:s,0:s));
A(sum(A,2) > s,:) = [];
A = 2 * A;

% leading order terms, for GRE
B = [2, 0; ...
     0, 2; ...
     1, 1];

n_train_MRE = size(A,1);
n_train_GRE = ((2^d) * factorial(d))^d;
n_train_SPRE = size(A,1) + 1; % plus one for LOOCV

% training inputs
X_MRE = (1/3) * (0.5 * ones(n_train_MRE,d) + 0.5 * rand(n_train_MRE,d)); 
X_GRE = (1/3) * (0.5 * ones(n_train_GRE,d) + 0.5 * rand(n_train_GRE,d)); 
X_SPRE = (1/3) * (0.5 * ones(n_train_SPRE,d) + 0.5 * rand(n_train_SPRE,d)); 

% training outputs
Y_MRE = f(X_MRE); 
Y_GRE = f(X_GRE); 
Y_SPRE = f(X_SPRE); 

% plotting limits
z_mean = mean([Y_MRE;Y_GRE;Y_SPRE]);
z_range = range([Y_MRE;Y_GRE;Y_SPRE]);
z_limits = [z_mean - 2*z_range, z_mean + 2*z_range];

%% MRE
out_MRE = MRE(A,X_MRE,Y_MRE);

%% GRE

k_name = "Matern3/2";

% GRE 
out_GRE = SPRE_opt(zeros(1,d),X_GRE,Y_GRE,{B,k_name});
x_opt = out_GRE.x;
out_GRE = SPRE(zeros(1,d),X_GRE,Y_GRE,x_opt,{B,k_name});

%% SPRE

k_name = "white";

% SPRE 
out_SPRE = SPRE_opt(A,X_SPRE,Y_SPRE,k_name);
x_opt = out_SPRE.x;
out_SPRE = SPRE(A,X_SPRE,Y_SPRE,x_opt,k_name);

%% plotting
nexttile
scatter3(0,0,f0,'pentagram'); hold on;
scatter3(0,0,out_MRE.mu,'^'); 
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
zlabel('$f(\mathbf{x})$','Interpreter','latex')
plot3drop(X_MRE(:,1),X_MRE(:,2),Y_MRE,z_limits,'ko')
title('MRE')

nexttile
scatter3(0,0,f0,'pentagram'); hold on;
scatter3(0,0,out_GRE.mu,'^');
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
zlabel('$f(\mathbf{x})$','Interpreter','latex')
plot3([0,0],[0,0],[out_GRE.mu - sqrt(out_GRE.var),out_GRE.mu + sqrt(out_GRE.var)])
plot3drop(X_GRE(:,1),X_GRE(:,2),Y_GRE,z_limits,'ko'); 
title('GRE')

nexttile
scatter3(0,0,f0,'pentagram'); hold on;
scatter3(0,0,out_SPRE.mu,'^'); 
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
zlabel('$f(\mathbf{x})$','Interpreter','latex')
plot3([0,0],[0,0],[out_SPRE.mu - sqrt(out_SPRE.var),out_SPRE.mu + sqrt(out_SPRE.var)])
plot3drop(X_SPRE(:,1),X_SPRE(:,2),Y_SPRE,z_limits,'ko')
title('SPRE')
legend({'$f(\mathbf{0})$', ...
        '$E[f(\mathbf{0}) | \{f(\mathbf{x}) : \mathbf{x} \in X_n \}]$', ...
        '$V[f(\mathbf{0}) | \{f(\mathbf{x}) : \mathbf{x} \in X_n \}]^{1/2}$', ...
        '$\{f(\mathbf{x}) : \mathbf{x} \in X_n$\}'},'Interpreter','latex')

fontsize("increase")





function out = Riemann_Sums(X,s)
% Numerical integration via the Riemann sum midpoint rule.

[n,d] = size(X); % dimension of the integrand
out = zeros(n,1);

% construct the integrand
% want the integrand to be in C^(2s+2)
syms z
phi = abs(z-0.5); % C^0
for i = 1:((2*s)+2)
    phi = int(phi,z); % C^i
end
phi = matlabFunction(phi,'Vars',z);
z_grid = linspace(0,d,100);
phi = @(z) phi(z) / max(abs(phi(z_grid))); % normalisation
T = sym("t",[1,d]);
g = 1 + phi(sum(T)/d); % C^(2s+2)

if isequaln(X,zeros(1,d))

    % true integral
    disp('Performing symbolic integration')
    gT = g;
    for i = 1:d
        disp(['Dimension ',num2str(i,'%u'),' of ',num2str(d,'%u'),'...'])
        gT = int(gT,T(i),0,1);
        disp('... done!')
    end
    out = eval(gT);

else

    g = matlabFunction(g,'Vars',T);

    if d == 1
        g = @(T) g(T); 
    elseif d == 2
        g = @(T) g(T(:,1),T(:,2)); 
    elseif d == 3
        g = @(T) g(T(:,1),T(:,2),T(:,3)); 
    end

    for j = 1:n

        x = X(j,:); % requested bin widths

        parfor i = 1:d
        
            % number of bins
            num_bins(i) = ceil(1/x(i)) - 1;
        
            % actual bin widths
            bin_width(i) = 1 / num_bins(i);
            
            % univariate bin centres
            centres{i} = linspace(0,1,num_bins(i) + 1);
            centres{i}(end) = [];
            centres{i} = centres{i} + 0.5 * centres{i}(2);
        
        end
        
        % multivariate bin centres
        T = table2array(combinations(centres{:}));

        % Riemann sum estimator
        out(j) = prod(bin_width) * sum(g(T));
    end

end

end


function plot3drop(X,Y,Z,z_limits,linestyle)

scatter3(X,Y,Z,linestyle);
hold on;
for i = 1:length(X)
    plot3([X(i),X(i)],[Y(i),Y(i)],[z_limits(1),Z(i)],'k:')
end
zlim(z_limits)

end