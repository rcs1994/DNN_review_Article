function show2Dproj(coordinates, elements, uh, u, varargin)
% show2Dproj  2D projected plots of uh and error |u-uh| on triangular mesh
%
% Usage:
%   show2Dproj(coordinates,elements,uh,u)
%   show2Dproj(...,'Clim',[min max],'BarLabel','uh','Colormap','jet','FontSize',14)

p = inputParser;
addParameter(p,'Clim',[], @(x)isnumeric(x) && (isempty(x) || numel(x)==2));
addParameter(p,'BarLabel','', @(x)ischar(x) || isstring(x));
addParameter(p,'Colormap','jet');
addParameter(p,'FontSize',14);
parse(p,varargin{:});
opts = p.Results;

uh = uh(:);
u  = u(:);

% % adjust element indexing if zero-based
% if min(elements(:)) == 0
%     elements = elements + 1;
% end

%% ---- Figure 1: uh ----
if isempty(opts.Clim)
    clim = [min(uh(:)) max(uh(:))];
else
    clim = opts.Clim;
end

fig1 = figure;
colormap(fig1, opts.Colormap);

patch('Faces',elements,'Vertices',coordinates, ...
      'FaceVertexCData',uh, ...
      'FaceColor','interp','EdgeColor','none');

axis equal
axis off
xlim([min(coordinates(:,1)) max(coordinates(:,1))])
ylim([min(coordinates(:,2)) max(coordinates(:,2))])
%title('Approximate solution u_h', 'FontSize', opts.FontSize)

caxis(clim)
cb = colorbar('Location','eastoutside','FontSize',opts.FontSize);
if ~isempty(opts.BarLabel)
    cb.Label.String = char(opts.BarLabel);
    cb.Label.FontSize = opts.FontSize;
end

% % overlay mesh lines (interior edges)
% hold on
% meshColor = [0.5 0.5 0.5];   % dark gray (change as you like)
% meshWidth = 0.35;               % thin interior lines
% triplot(elements, coordinates(:,1), coordinates(:,2), 'Color', meshColor, 'LineWidth', meshWidth);

% highlig

set(fig1,'Color','w');
saveas(gcf, 'Approximate_Solution.png');

%% ---- Figure 2: |u - uh| ----
err = abs(u - uh);

fig2 = figure;
colormap(fig2, opts.Colormap);

patch('Faces',elements,'Vertices',coordinates, ...
      'FaceVertexCData',err, ...
      'FaceColor','interp','EdgeColor','none');

axis equal
axis off
xlim([min(coordinates(:,1)) max(coordinates(:,1))])
ylim([min(coordinates(:,2)) max(coordinates(:,2))])
%title('Absolute error |u - u_h|', 'FontSize', opts.FontSize)

caxis([min(err(:)) max(err(:))])
cb2 = colorbar('Location','eastoutside','FontSize',opts.FontSize);
%cb2.Label.String = '|u - u_h|';
cb2.Label.FontSize = opts.FontSize;

% % overlay mesh lines (interior edges)
% hold on
% meshColor = [0.5 0.5 0.5];   % dark gray (change as you like)
% meshWidth = 0.35;               % thin interior lines
% triplot(elements, coordinates(:,1), coordinates(:,2), 'Color', meshColor, 'LineWidth', meshWidth);

set(fig2,'Color','w');
% Save figure 3
saveas(gcf, 'Error.png');
end
