function show(coordinates,elements,uh,u)

subplot(1,2,1)
trisurf(elements,coordinates(:,1),coordinates(:,2),uh','facecolor','interp')
subplot(1,2,2)
trisurf(elements,coordinates(:,1),coordinates(:,2),u','facecolor','interp')
