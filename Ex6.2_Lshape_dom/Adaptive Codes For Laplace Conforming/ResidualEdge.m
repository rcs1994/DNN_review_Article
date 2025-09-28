function etaE=ResidualEdge(uxh,uyh,Elem,Coord,n2ed,ed2el,Nb)
   etaE=zeros(size(Elem,1),1);
   nrEdges=size(ed2el,1);
   
   for i=1:nrEdges,
       Elem1=ed2el(i,3);
       Elem2=ed2el(i,4);
       if Elem2~=0
          Node1=ed2el(i,1);
          Node2=ed2el(i,2);
          Coord1 = Coord(Node1,:);
          Coord2 = Coord(Node2,:);
          h_e=norm(Coord1-Coord2);
          cU=(Elem1-1)*3; aU=(Elem2-1)*3;
          uxh_E1=uxh(cU+1); uxh_E2=uxh(aU+1);
          uyh_E1=uyh(cU+1); uyh_E2=uyh(aU+1);       
          Jum=[uxh_E1-uxh_E2  uyh_E1-uyh_E2];
          Normal_ed=[Coord2(2)-Coord1(2) Coord1(1)-Coord2(1)]/h_e;
          
          Jum1=(h_e^2*(Normal_ed*Jum')^2);
          
          etaE(Elem1)=etaE(Elem1)+Jum1;
          etaE(Elem2)=etaE(Elem2)+Jum1;
       end
   end
   etaE=etaE/2;
   
   if (~isempty(Nb))
     for j=1:size(Nb,1)
          Node1=Nb(j,1);
          Node2=Nb(j,2); Elem1=ed2el(n2ed(Node1,Node2),3);
          Coord1 = Coord(Node1,:);
          Coord2 = Coord(Node2,:);
          h_e=norm(Coord1-Coord2);
          cU=(Elem1-1)*3; 
          [ux,uy]=uxe((Coord1+Coord2)/2);
          uxh_E1=uxh(cU+1); 
          uyh_E1=uyh(cU+1);       
          Jum=[uxh_E1-ux  uyh_E1-uy];
          Normal_ed=[Coord2(2)-Coord1(2) Coord1(1)-Coord2(1)]/h_e;
          
          Jum1= h_e^2*(Normal_ed*Jum')^2;
          
          etaE(Elem1)=etaE(Elem1)+Jum1;
       end
   end
   
   etaE=etaE.^(1/2);
   
   
   

   
  