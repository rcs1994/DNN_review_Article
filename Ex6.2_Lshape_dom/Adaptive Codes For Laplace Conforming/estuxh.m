function [uxh,uyh]=estuxh(x,n4e,c4n,grad4e)

   for i=1:size(n4e,1)
       curNodes = n4e(i,:);
       curCoords = c4n(curNodes,:); cU=(i-1)*3;
       P1=curCoords(1,:); P2=curCoords(2,:); P3=curCoords(3,:);
       curGrad=grad4e(:,:,i);
       L1=curGrad(1,:); L2=curGrad(2,:); L3=curGrad(3,:);
       cx=x(curNodes); 
       uxh(1,cU+1)=cx(1)*L1(1)+cx(2)*L2(1)+cx(3)*L3(1);          
       uxh(1,cU+2)=uxh(1,cU+1);                                  
       uxh(1,cU+3)=uxh(1,cU+1);                                  
       
       uyh(1,cU+1)=cx(1)*L1(2)+cx(2)*L2(2)+cx(3)*L3(2);          
       uyh(1,cU+2)=uyh(1,cU+1);                                  
       uyh(1,cU+3)=uyh(1,cU+1);                                  
      
   end
