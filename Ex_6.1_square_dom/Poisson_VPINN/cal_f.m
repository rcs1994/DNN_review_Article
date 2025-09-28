syms x y r

u1=(sin(pi*x)).^2*sin(pi*y)*cos(pi*y);
u2=-(sin(pi*y)).^2*sin(pi*x)*cos(pi*x);
u1x=diff(u1,x);u1y=diff(u1,y);
u1xx=diff(u1x,x);u1yy=diff(u1y,y);
u2x=diff(u2,x);u2y=diff(u2,y);
u2xx=diff(u2x,x);u2yy=diff(u2y,y);
p=sin(2*pi*x)*sin(2*pi*y); 
px=diff(p,x);py=diff(p,y);
f1=-r*(u1xx+u1yy)+px+u1*u1x+u2*u1y
f2=-r*(u2xx+u2yy)+py+u1*u2x+u2*u2y
