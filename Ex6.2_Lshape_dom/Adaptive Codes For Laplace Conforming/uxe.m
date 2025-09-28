function [uxv,uyv]=uxe(z)
x=z(1); y=z(2);

t=cart2pol(x,y); if t<0, t=t+2*pi; end
r=sqrt(x^2+y^2);

if r==0
    uxv=0; uyv=0;
else
 u_r=2/3*r^(-1/3)*sin(2*t/3);
 u_t=r^(2/3)*cos(2*t/3)*2/3;
    
 uxv=cos(t)*u_r-sin(t)/r*u_t;
 uyv=sin(t)*u_r+cos(t)/r*u_t;
end

% uxv=y;
% uyv=x;