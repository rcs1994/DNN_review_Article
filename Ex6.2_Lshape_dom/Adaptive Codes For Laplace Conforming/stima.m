function M=stima(mk,grads)

% barycentic coordinates of current element
L1=grads(1,:);
L2=grads(2,:);
L3=grads(3,:);

% element stiffness matrix
M=mk*[L1*L1' L1*L2' L1*L3'
      L2*L1' L2*L2' L3*L2'
      L3*L1' L3*L2' L3*L3'];
   

