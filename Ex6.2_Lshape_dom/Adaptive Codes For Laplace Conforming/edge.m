
function [n2el,n2ed,ed2el]=edge(elem,node)
% to enumerate number of edges
n2el=sparse(size(node,1),size(node,1));
for j=1:size(elem,1)
    n2el(elem(j,:),elem(j,[2 3 1]))=...
      n2el(elem(j,:),elem(j,[2 3 1]))+j*eye(3,3);
end

B=n2el+n2el';
[i,j]=find(triu(B));
n2ed=sparse(i,j,1:size(i,1),size(node,1),size(node,1));
n2ed=n2ed+n2ed';
%noedges=size(i,1);

% to generate elem of edge
ed2el=zeros(size(i,1),4);
for m = 1:size(elem,1)
  for k = 1:3
   
    p = n2ed(elem(m,k),elem(m,rem(k,3)+1)); 
    
    if ed2el(p,1)==0  
     
       ed2el(p,:)=[elem(m,k) elem(m,rem(k,3)+1) ...
                     n2el(elem(m,k),elem(m,rem(k,3)+1))...
                     n2el(elem(m,rem(k,3)+1),elem(m,k))];            
          
    end
  end
end
  
 



