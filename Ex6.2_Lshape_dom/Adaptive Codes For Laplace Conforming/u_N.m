function unv=u_N(P1,P2,NN)
x1=P1(1);y1=P1(2);
x2=P2(1);y2=P2(2);
MP=(P1+P2)/2;

x=MP(1); y=MP(2);


[ux,uy]=uxe([x y]);  

unv=ux*NN(1)+uy*NN(2);

