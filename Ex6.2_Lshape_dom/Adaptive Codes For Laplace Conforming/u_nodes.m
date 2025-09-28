function u=u_nodes(coordinates)
for j=1:size(coordinates,1),
    curcoords=coordinates(j,:);
    u(j,1)=ue(curcoords);
end