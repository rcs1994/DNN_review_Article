function [Coord,Elem,Nb,Db]=bisectionedge(Coord,Elem,n2el,n2ed,Nb,Db,nrEdges,eta,theta)
NT=size(Elem,1); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
total=sum(eta); [temp,ix]=sort(-eta);
current=0; marker=zeros(nrEdges,1);
for t=1:NT
    if (current>theta *total),break,end
    index=1; ct=ix(t);
    while (index==1)
        base=n2ed(Elem(ct,2),Elem(ct,3));
        if (marker(base)>0),index=0;
        else
            current=current+eta(ct);
            N=size(Coord,1)+1;
            marker(base)=N;
            Coord(N,:)=mean(Coord(Elem(ct,[2,3]),:));
            ct=n2el(Elem(ct,3),Elem(ct,2));
            if (ct==0),index=0;end
        end
    end
end


for t=1:NT
    base=n2ed(Elem(t,2),Elem(t,3));
    if (marker(base)>0)
        p=[Elem(t,:),marker(base)];
        Elem=divide(Elem,t,p);
        left=n2ed(p(1),p(2));  right=n2ed(p(3),p(1));
        if (marker(right)>0)
           Elem=divide(Elem,size(Elem,1),[p(4),p(3),p(1),marker(right)]);
        end
        if (marker(left)>0)
           Elem=divide(Elem,t,[p(4),p(1),p(2),marker(left)]);
        end
    end
end

%%%% Boundary Edges
nb=size(Db,1);
if (nb>0)
    for i=1:nb,
        base=n2ed(Db(i,1),Db(i,2));
        if (marker(base)>0)
            p=[Db(i,1) marker(base)  Db(i,2)];
            Db(i,:)=[p(1) p(2)];
            Db(size(Db,1)+1,:)=[p(2) p(3)];
        end
    end
end

nb=size(Nb,1);
if (nb>0)
    for i=1:nb,
         base=n2ed(Nb(i,1),Nb(i,2));
        if (marker(base)>0)
            p=[Nb(i,1) marker(base)  Nb(i,2)];
            Nb(i,:)=[p(1) p(2)];
            Nb(size(Nb,1)+1,:)=[p(2) p(3)];
        end
    end
end

