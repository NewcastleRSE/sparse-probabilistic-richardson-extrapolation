%% Pointwise addition for a collection of arrays stored in a cell

function out = cellsum(in)

    d = length(in);
    out = in{1};
    for i = 2:d
        out = out + in{i};
    end
    
end