function [H1e,L2e,H1_rel_err, L2_rel_err]=Err(Coord,Elem,grad4e,area4e, uh)

 H1e=0; L2e=0; L2_norm_u=0; H1_norm_u = 0;

for j=1:size(Elem,1),
    curnodes=Elem(j,:);
    curcoords=Coord(curnodes,:);
    Cuh=uh(curnodes);
    P1=curcoords(1,:); P2=curcoords(2,:); P3=curcoords(3,:);
    mp12=1/2*(P1+P2); mp13=1/2*(P1+P3); mp23=1/2*(P3+P2);
    cg=1/3*(P1+P2+P3);
    
    uhmp12=1/2*(Cuh(1)+Cuh(2)); uhmp13=1/2*(Cuh(1)+Cuh(3));
    uhmp23=1/2*(Cuh(3)+Cuh(2)); uhcg=1/3*(Cuh(1)+Cuh(2)+Cuh(3));         
    
    curGrads=grad4e(:,:,j); 
    L1=curGrads(1,:); L2=curGrads(2,:); L3=curGrads(3,:); 
    mk=area4e(j);
    
    uxh=Cuh(1)*L1(1)+Cuh(2)*L2(1)+Cuh(3)*L3(1);
    uyh=Cuh(1)*L1(2)+Cuh(2)*L2(2)+Cuh(3)*L3(2);

    [ux1 uy1]=uxe(mp23);
    [ux2 uy2]=uxe(mp13);
    [ux3 uy3]=uxe(mp12);
    
    H1e = H1e+mk/3*((ux1-uxh)^2+(ux2-uxh)^2+(ux3-uxh)^2+(uy1-uyh)^2+(uy2-uyh)^2+(uy3-uyh)^2)+...
        +mk/60*(8*(ue(mp12)-uhmp12)^2+8*(ue(mp13)-uhmp13)^2+8*(ue(mp23)-uhmp23)^2+...
            3*(ue(P1)-Cuh(1))^2+3*(ue(P2)-Cuh(2))^2+3*(ue(P3)-Cuh(3))^2+27*(ue(cg)-uhcg)^2);

    H1_norm_u = H1_norm_u + mk/3*((ux1)^2+(ux2)^2+(ux3)^2+(uy1)^2+(uy2)^2+(uy3)^2)+...
                 mk/60*(8*(ue(mp12))^2+8*(ue(mp13))^2+8*(ue(mp23))^2+...
                  3*(ue(P1))^2+3*(ue(P2))^2+3*(ue(P3))^2+27*(ue(cg)^2));



    L2e=L2e+mk/60*(8*(ue(mp12)-uhmp12)^2+8*(ue(mp13)-uhmp13)^2+8*(ue(mp23)-uhmp23)^2+...
            3*(ue(P1)-Cuh(1))^2+3*(ue(P2)-Cuh(2))^2+3*(ue(P3)-Cuh(3))^2+27*(ue(cg)-uhcg)^2);

    L2_norm_u = L2_norm_u + mk/60*(8*(ue(mp12))^2+8*(ue(mp13))^2+8*(ue(mp23))^2+...
            3*(ue(P1))^2+3*(ue(P2))^2+3*(ue(P3))^2+27*(ue(cg)^2));
    
   end
   H1e=sqrt(H1e); L2e=sqrt(L2e); L2_norm_u = sqrt(L2_norm_u); L2_rel_err = L2e/L2_norm_u; H1_norm_u = sqrt(H1_norm_u); H1_rel_err = H1e/H1_norm_u;
end



