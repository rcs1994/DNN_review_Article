function uv=ue(z)
x=z(1); y=z(2);

t=cart2pol(x,y); if t<0, t=t+2*pi; end
r=sqrt(x^2+y^2);

uv=r^(2/3)*sin(2*t/3);

%uv=x*y;