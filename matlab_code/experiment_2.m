%% Empirically confirm convergence acceleration is achieved

set(groot,'defaultAxesTickLabelInterpreter','latex'); 
set(groot,'defaultLegendInterpreter','latex');

clear all
rng(1)
addpath(genpath("helper_functions"))

% plotting
RGB = orderedcolors("gem"); % colour palette
HEX = rgb2hex(RGB); % hex colours
figure()
tiledlayout(3,4)

for d = 1:3 % data dimension, can be in {1,2,3}
    for s = 0:1 % the integrand is C^(2s+2)

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
        g = 1 + phi(sum(abs(T))/d); % C^(2s+2) (the abs has no effect but helps the symbolic solver

        % true integral
        disp('Performing symbolic integration')
        gT = g;
        for i = 1:d
            disp(['Dimension ',num2str(i,'%u'),' of ',num2str(d,'%u'),'...'])
            gT = int(gT,T(i),0,1);
            disp('... done!')
        end
        f0 = eval(gT);

        g = matlabFunction(g,'Vars',T);
        if d == 1
            g = @(T) g(T); 
        elseif d == 2
            g = @(T) g(T(:,1),T(:,2)); 
        elseif d == 3
            g = @(T) g(T(:,1),T(:,2),T(:,3)); 
        end

        % target
        f = @(X) Riemann_Sums(X,g); % Riemann sums numerical method

        % sparse basis, for SPRE
        if d == 1
            A = table2array(combinations(0:s));
        elseif d == 2
            A = table2array(combinations(0:s,0:s));
        elseif d == 3
            A = table2array(combinations(0:s,0:s,0:s));
        end
        A(sum(A,2) > s,:) = [];
        A = 2 * A;

        % leading order terms, for GRE
        if d == 1
            B = [2];
        elseif d == 2
            B = [2, 0; ...
                 0, 2; ...
                 1, 1];
        elseif d == 3
            B = [2, 0, 0; ...
                 1, 1, 0; ...
                 1, 0, 1; ...
                 0, 1, 1; ...
                 0, 2, 0; ...
                 0, 0, 2];
        end
        
        % reference design set
        if d == 1
            X0 = (1/2) * 1./[1;2;3;4];
        elseif d == 2
            X0 = (1/2) * 1./[1,1; 1,2; 2,1; 2,2; 1,3; 3,1];
        elseif d == 3
            X0 = (1/2) * 1./[1,1,1; 2,1,1; 1,2,1; 1,1,2; 2,2,1; 2,1,2; 1,2,2; 2,2,2];
        end

        h_min_1D = 2^(-17); % smallest bin width for a d = 1 task
        h_min_dD = h_min_1D^(1/d); % smallest bin width for a d-dimensional task
        h_vals = 2.^(-19:0); % values of scaling constant
        h_vals(h_vals < h_min_dD) = []; % remove values that are too small
        
        H = length(h_vals);
        
        abs_errors_NN = zeros(1,H);
        abs_errors_MRE = zeros(1,H);
        abs_errors_GRE = zeros(1,H);
        rel_errors_GRE = zeros(1,H);
        abs_errors_SPRE = zeros(1,H);
        rel_errors_SPRE = zeros(1,H);
        parfor i = 1:H
        
            h = h_vals(i);
            X = h * X0; % training inputs
            Y = f(X); % training outputs
        
            %% Nearest neighbour
            idx = knnsearch(X,zeros(1,d),'K',1); % find datum closest to 0
            abs_errors_NN(i) = abs(Y(idx) - f0);
        
            %% MRE
            out_MRE = MRE(A,X,Y);
        
            % absolute error
            abs_errors_MRE(i) = abs(out_MRE.mu - f0);
        
            %% GRE

            k_name = "Matern3/2";

            % optimise kernel parameters for GRE
            out_GRE = SPRE_opt(zeros(1,d),X,Y,{B,k_name});
            x_opt = out_GRE.x;

            % GRE 
            out_GRE = SPRE(zeros(1,d),X,Y,x_opt,{B,k_name});

            % absolute error
            abs_errors_GRE(i) = abs(out_GRE.mu - f0);

            % relative error
            rel_errors_GRE(i) = (out_GRE.mu - f0) / sqrt(out_GRE.var);

            %% SPRE

            k_name = "white";

            % optimise kernel parameters for SPRE
            out_SPRE = SPRE_opt(A,X,Y,k_name);
            x_opt = out_SPRE.x;

            % SPRE 
            out_SPRE = SPRE(A,X,Y,x_opt,k_name);

            % absolute error
            abs_errors_SPRE(i) = abs(out_SPRE.mu - f0);

            % relative error
            rel_errors_SPRE(i) = (out_SPRE.mu - f0) / sqrt(out_SPRE.var);
        
        end

        % minimum achievable error is machine precision
        abs_errors_NN = max(eps,abs_errors_NN);
        abs_errors_MRE = max(eps,abs_errors_MRE);
        abs_errors_GRE = max(eps,abs_errors_GRE);
        abs_errors_SPRE = max(eps,abs_errors_SPRE);
        
        % plotting
        nexttile
        loglog(h_vals,abs_errors_NN,'-square','Color',HEX(5)); hold on
        loglog(h_vals,abs_errors_MRE,'-x','Color',HEX(2))
        loglog(h_vals,abs_errors_GRE,'-^','Color',HEX(1))
        loglog(h_vals,abs_errors_SPRE,'-o','Color','k')
        if d == 3
            xlabel('$h$','Interpreter','latex')
        end
        ylabel('absolute error','Interpreter','latex')
        if (d == 3) && (s == 1)
            legend({'Baseline','MRE','GRE','SPRE'},"Location","southeast",'Interpreter','latex')
        end
        grid on
        axis square
        title(['$d=',num2str(d,'%u'),'$, $s=',num2str(s,'%u'),'$'],'Interpreter','latex')
        nexttile
        tmp1 = semilogx(h_vals,rel_errors_GRE,'-^','Color',HEX(1)); hold on
        tmp2 = semilogx(h_vals,rel_errors_SPRE,'-o','Color','k');
        shade_background(@(x,y) normpdf(y))
        delete(tmp1); delete(tmp2);
        semilogx(h_vals,rel_errors_GRE,'-^','Color',HEX(1)); hold on
        semilogx(h_vals,rel_errors_SPRE,'-o','Color','k');
        ylabel('relative error','Interpreter','latex')
        if d == 3
            xlabel('$h$','Interpreter','latex')
        end
        grid on
        axis square
        title(['$d=',num2str(d,'%u'),'$, $s=',num2str(s,'%u'),'$'],'Interpreter','latex')

    end
end

fontsize("increase")
fontsize("increase")
fontsize("increase")
fontsize("increase")
set(gcf,'Color','white')


function out = Riemann_Sums(X,g)
% Numerical integration via the Riemann sum midpoint rule.
% g is the function to be integrated

[n,d] = size(X); % dimension of the integrand
out = zeros(n,1);

for j = 1:n

    x = X(j,:); % requested bin widths

    centres = cell(d,1);
    for i = 1:d
        
        % number of bins
        num_bins(i) = ceil(1/x(i));
    
        % actual bin widths
        bin_width(i) = 1/num_bins(i);
        
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

