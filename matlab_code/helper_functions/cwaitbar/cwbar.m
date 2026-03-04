function cwbar(p)
% CWBAR - a command line waitbar
% 
% Usage:
%   To initalize the bar:
%       cwbar(startmsg); % where startmsg is any string not starting 
%                        % with '!', startmsg will appear before the bar
%   To update the bar in a loop:
%       cwbar(p);        % where p is a number from 0 to 1
%   To finalize the bar
%       cwbar(endmsg);   % where endmsg is any string starting with '!'.
%                        % endmsg(2:end) will appear after the bar followed
%                        % by a new line
%
% Date  : 09/10/2021
% Author: Amal Joy
%
    N = 15; % The length of the wait bar
    if isnumeric(p)
        assert(p >= 0 && p <= 1, 'cwbar: p a non-negative number less than 1');
        lit = floor(N*p);
        fprintf([
            repmat('\b', 1, N + 7),             ...
            sprintf('%3d%%%%', floor(100*p)),   ...
            ' [',                               ...
            repmat('=', 1, lit),                ...
            repmat('>', 1, p~=1),               ...
            repmat('.', 1, N-lit-1),            ...
            ']'                                 ...
        ]);
    elseif isstring(p) || ischar(p)
        p = char(p);
        if p(1) == '!'
            cwbar(1);
            fprintf([' ', p(2:end), '\n']);
        else
            fprintf([p, repmat(' ', 1, N + 7)]);
            cwbar(0);
        end
    else
        error('Unsupported type. Supported types are string and num');
    end
end