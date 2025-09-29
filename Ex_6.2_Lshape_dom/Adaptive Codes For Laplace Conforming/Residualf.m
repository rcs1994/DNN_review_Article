function etaf=Residualf(Elem,Coord,area4e);

for j=1:size(Elem,1),
    curnodes=Elem(j,:);
    curcoords=Coord(curnodes,:);
 
    P1=curcoords(1,:); P2=curcoords(2,:); P3=curcoords(3,:);
    mk=area4e(j); mp23=1/2*(P2+P3);  mp13=1/2*(P1+P3);   mp12=1/2*(P1+P2); % midpoints of edges
     
    L2f=mk/3*(f(mp23)^2+f(mp13)^2+f(mp12)^2);
    
    etaf(j,1)=sqrt(mk*L2f);
end