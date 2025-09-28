function elem=divide(elem,t,p)
elem(size(elem,1)+1,:)=[p(4) p(3) p(1)];
elem(t,:)=[p(4) p(1) p(2)];