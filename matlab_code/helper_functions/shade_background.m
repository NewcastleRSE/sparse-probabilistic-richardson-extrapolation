%% Helper function to plot a shaded background

function shade_background(p)

lim_x = xlim;
lim_y = ylim;

x1_vals = linspace(lim_x(1),lim_x(2),1000);
x2_vals = linspace(lim_y(1),lim_y(2),1000);

f_vals = zeros(length(x1_vals),length(x2_vals));
for i = 1:length(x1_vals)
    for j = 1:length(x2_vals)
        f_vals(j,i) = p(x1_vals(i),x2_vals(j));
    end
end
faintness_factor = 0.5;
f_vals = faintness_factor * f_vals / max(f_vals(:));

colormap('gray')
image(lim_x,lim_y,255*(1-f_vals));
set(gca,'YDir','normal') 

% put into the background
set(gca,'layer','top');
%h = get(gca,'Children');
%h = [h(2:end); h(1)];
%set(gca,'Children',h)

xlim(lim_x)
ylim(lim_y)

