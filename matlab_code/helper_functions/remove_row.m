%% Remove rows from an array

function out = remove_row(in,index)

    out = in;
    out(index,:) = [];
    
end