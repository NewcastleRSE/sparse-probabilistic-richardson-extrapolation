% Sequential experimental design

set(groot,'defaultAxesTickLabelInterpreter','latex'); 
set(groot,'defaultLegendInterpreter','latex');

clear all
addpath(genpath("helper_functions"))

% plotting
RGB = orderedcolors("gem"); % colour palette
HEX = rgb2hex(RGB); % hex colours
figure()
ax = tight_subplot(2,4,0.07);

% realise the white noise process
rng(0)
n_noise = 10000;
noise = randn(n_noise);
gd = linspace(0,1,n_noise); % 1D grid
noise_process = @(X) interp2(gd,gd,noise,X(:,1),X(:,2));

% true data-generating function
% designed so true maximal A is {(0,0),(1,0),(0,1),(1,1),(2,0)}
d = 2; % dimension
R = @(X) 0.0001 * X(:,1) .* X(:,2) .* noise_process(X); % residual
f = @(X) 1 + X(:,1) - 2 * X(:,2) + 3 * X(:,1).^2 + R(X);

f0 = 1; % true quantity of interest

% cost function
c = @(X) (X(:,1) .* X(:,2)).^(-1);

% budget
C = 2;

% number of iterations
its = 8;

% initial design set
X = cell(its+1,1);
X{1} = 1./[1,1;
           2,1;
           1,2];
X{1} = X{1} - 0.01 * rand(size(X{1})); % jitter to avoid singular matrix in derivative calculation for optimisation

A = cell(its,1);
sigma = cell(its,1);
for i = 1:its

    % evaluate data
    Y = f(X{i});
    
    % learn (k,A) using SPRE
    k_name = "Matern1/2";
    max_order = inf;
    out = SPRE_stepwise(X{i},Y,k_name,max_order);
    x = out.x; % estimated kernel parameter (sign unconstrained)
    sigma{i} = sqrt(softplus(x(1)));
    A{i} = out.A; % estimated index set
    
    % experimental design
    n_attempts = 1000; % number of candidate designs to examine
    X_new = experimental_design(A{i},X{i},x,k_name,c,C,n_attempts,i);
    X{i+1} = [X{i}; X_new];

    % generate plot title in latex
    tit = '$A = \{';
    for j = 1:size(A{i},1)
        tit = [tit,'(',num2str(A{i}(j,1),'%u'),',',num2str(A{i}(j,2),'%u'),')'];
        if j == size(A{i},1)
            tit = [tit,'\}$'];
        else
            tit = [tit,','];
        end
    end

    % plotting
    axes(ax(i))
    scatter(X{i}(:,1),X{i}(:,2),'ko'); hold on;
    scatter(X_new(:,1),X_new(:,2),'bo','filled');
    title(tit,'Interpreter','latex')
    xlabel('$x_1$','Interpreter','latex')
    ylabel('$x_2$','Interpreter','latex')
    xlim([0,1])
    ylim([0,1])
    xticks([0,1])
    yticks([0,1])
    text(0.1,0.1,['$\sigma = ',num2str(sigma{i},2),'$'],'Interpreter','latex')
    text(0.1,0.25,['iter. $',num2str(i,'%u'),'$'],'Interpreter','latex')
    axis square
    box on
    if i == its
        legend({'previously selected','new design set'})
    end

end

set(gcf,'Color','white')
fontsize("increase")
fontsize("increase")



% experimental design by stochastic search
function out = experimental_design(A,X,x,k_name,c,C,n_attempts,seed)

[~,d] = size(X);

X_new = cell(n_attempts,1);
variance = zeros(n_attempts,1);
parfor i = 1:n_attempts

    rng((seed-1)*n_attempts + i) % ensure reproducility in a parfor loop

    % generate a new design
    X_new{i} = ones(0,d);
    not_saturated = true;
    trivial = true;
    while not_saturated || trivial
        x_rand = rand(1,d);
        if c([X_new{i}; x_rand]) < C
            X_new{i} = [X_new{i}; x_rand];
            trivial = false;
        else
            not_saturated = false;
        end
    end

    % assess new design
    X_total = [X; X_new{i}]; % total design set
    X_total = unique(X_total,'rows');
    n_total = size(X_total,1); % total number of design points
    grad = false; % don't compute gradient
    out = SPRE(A,X_total,zeros(n_total,1),x,k_name,grad);
    variance(i) = out.var;

end

% return the best design found
[~,idx] = min(variance);
out = X_new{idx};

end