%% White noise kernel

function out = white(X1,X2)

[n1,~] = size(X1);
[n2,~] = size(X2);

out = zeros(n1,n2);

[~,Locb] = ismember(X1,X2,"rows");

for i = 1:n1
    if Locb(i) ~= 0
        out(i,Locb(i)) = 1;
    end
end

end