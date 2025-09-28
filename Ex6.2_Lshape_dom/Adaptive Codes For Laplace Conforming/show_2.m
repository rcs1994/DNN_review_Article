function show_2(coordinates, elements, uh, u)
% Create three separate figures with 2D colormap visualizations and save them

% Calculate error
error = abs(u - uh);

% Figure 1: Approximate solution (uh)
figure(10)
trisurf(elements, coordinates(:,1), coordinates(:,2), zeros(size(uh)), uh, ...
    'EdgeColor', [0.5 0.5 0.5], 'FaceColor', 'interp'); % Gray edges for mesh visibility
view(2) % 2D view
colormap('jet')
colorbar
set(gca, 'XTick', [], 'YTick', [])
%title('Approximate Solution (u_h)')
axis equal tight

% Save figure 1
saveas(gcf, 'Approximate_Solution.png');

% Figure 2: Exact solution (u)
figure(20)
trisurf(elements, coordinates(:,1), coordinates(:,2), zeros(size(u)), u, ...
    'EdgeColor', 'none', 'FaceColor', 'interp');
view(2) % 2D view
colormap('jet')
colorbar
set(gca, 'XTick', [], 'YTick', [])
%title('Exact Solution (u)')
axis equal tight

% Save figure 2
saveas(gcf, 'Exact_Solution.png');

% Figure 3: Error |u - uh|
figure(30)
trisurf(elements, coordinates(:,1), coordinates(:,2), zeros(size(error)), error, ...
    'EdgeColor', 'none', 'FaceColor', 'interp');
view(2) % 2D view
colormap('jet')
colorbar
set(gca, 'XTick', [], 'YTick', [])
%title('Error |u - u_h|')
axis equal tight

% Save figure 3
saveas(gcf, 'Error.png');

end