function show_1(coordinates, elements, uh, u)
% show  3-panel 2D projection: uh, u, and |u-uh|
%
% Inputs:
%   coordinates : Nnodes x 2 array [x y]
%   elements    : Nelems x 3 array of triangle indices (0- or 1-based)
%   uh, u       : vectors (Nnodes x 1) nodal values
%
% Produces a single figure with 3 subplots:
%   [1] u_h, [2] u (exact), [3] |u-u_h|

% --- basic checks & normalize shapes ---
if size(coordinates,2) ~= 2
    error('coordinates must be N x 2');
end
if size(elements,2) ~= 3
    error('elements must be M x 3 triangles');
end

uh = uh(:);
u  = u(:);

if numel(uh) ~= size(coordinates,1) || numel(u) ~= size(coordinates,1)
    error('Length of uh and u must equal number of nodes (rows of coordinates).');
end

% convert zero-based elements (if any) to 1-based
if min(elements(:)) == 0
    elements = elements + 1;
end

% compute error
err = abs(u - uh);

% determine color limits
% use same color limits for uh and u for visual comparison
vmin_shared = min([uh; u]);
vmax_shared = max([uh; u]);

% if constant field, expand limits slightly to avoid warnings
if vmin_shared == vmax_shared
    vmin_shared = vmin_shared - 1e-8;
    vmax_shared = vmax_shared + 1e-8;
end

% error color limits
vmin_err = min(err);
vmax_err = max(err);
if vmin_err == vmax_err
    vmin_err = vmin_err - 1e-8;
    vmax_err = vmax_err + 1e-8;
end

% plotting parameters
fs = 14;           % font size
cmap = 'jet';      % colormap

% create figure with 3 subplots
figure('Color','w','Units','normalized','Position',[0.05 0.2 0.9 0.5]);

% ------------- panel 1: uh -------------
ax1 = subplot(1,3,1);
patch('Faces',elements,'Vertices',coordinates, ...
      'FaceVertexCData',uh, 'FaceColor','interp','EdgeColor','none', 'Parent', ax1);
axis(ax1,'equal'); axis(ax1,'off')
title(ax1,'u_h (approx)','FontSize',fs)
colormap(ax1, cmap);
caxis(ax1,[vmin_shared vmax_shared]);
cb1 = colorbar(ax1,'eastoutside','FontSize',fs);
cb1.Label.String = 'u_h';
cb1.Label.FontSize = fs;

% ------------- panel 2: u -------------
ax2 = subplot(1,3,2);
patch('Faces',elements,'Vertices',coordinates, ...
      'FaceVertexCData',u, 'FaceColor','interp','EdgeColor','none', 'Parent', ax2);
axis(ax2,'equal'); axis(ax2,'off')
title(ax2,'u (exact)','FontSize',fs)
colormap(ax2, cmap);
caxis(ax2,[vmin_shared vmax_shared]);
cb2 = colorbar(ax2,'eastoutside','FontSize',fs);
cb2.Label.String = 'u';
cb2.Label.FontSize = fs;

% ------------- panel 3: |u-uh| -------------
ax3 = subplot(1,3,3);
patch('Faces',elements,'Vertices',coordinates, ...
      'FaceVertexCData',err, 'FaceColor','interp','EdgeColor','none', 'Parent', ax3);
axis(ax3,'equal'); axis(ax3,'off')
title(ax3,'|u - u_h|','FontSize',fs)
colormap(ax3, cmap);
caxis(ax3,[vmin_err vmax_err]);
cb3 = colorbar(ax3,'eastoutside','FontSize',fs);
cb3.Label.String = '|u-u_h|';
cb3.Label.FontSize = fs;

% Improve spacing (optional)
% tight layout like behavior
drawnow;
% Add a tiny border around result area
for ax = [ax1, ax2, ax3]
    outerpos = get(ax,'OuterPosition');
    set(ax,'OuterPosition',outerpos .* [1 1 0.98 1]); %#ok<*MSET>
end

end
