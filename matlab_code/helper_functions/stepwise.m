%% Compute which high-order interactions to consider next

function out = stepwise(A,order)

out = [];
[n_models,d] = size(A);
for i = 1:n_models
    Ai = A(i,:);
    if sum(Ai) == (order-1) % ignore interactions of order-2 and lower
        for j = 1:d
            Aij = Ai; Aij(j) = Aij(j) + 1; % increment the jth entry in Ai
            out = [out; Aij];
        end
    end
end

% remove duplicate models
out = unique(out,'rows');

end