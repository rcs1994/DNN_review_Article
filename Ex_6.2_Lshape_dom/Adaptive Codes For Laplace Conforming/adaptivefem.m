 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEM for  - div . (u_x,u_y)  =  f  in \Omega
%                   (u_x, u_y). n = u_N on \partial\Omega_N
%                              u  = g  on \partial\Omega_D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;
tt=cputime;
[Coord,Elem,Nb,Db]=InitialMesh;

area4e = getArea4e(Coord,Elem);
grad4e = getGrad4e(Coord,Elem,area4e);
[n2el,n2ed,ed2el]=edge(Elem,Coord);

nrEdges=size(ed2el,1);
nrElems=size(Elem,1);
nrNodes=size(Coord,1);

A=sparse(size(Coord,1),size(Coord,1));
b=sparse(size(Coord,1),1);

for j=1:size(Elem,1),
    area=area4e(j);
    grads=grad4e(:,:,j);
    A(Elem(j,:),Elem(j,:))=A(Elem(j,:),Elem(j,:))+stima(area,grads);
end

for j=1:size(Elem,1),
    b(Elem(j,:))=b(Elem(j,:))+area4e(j)*f(sum(Coord(Elem(j,:),:))/3)/3;
end

if (~isempty(Nb))
    for j=1:size(Nb,1)
        led=norm(Coord(Nb(j,1),:)-Coord(Nb(j,2),:));
        NormalV=[Coord(Nb(j,2),2)-Coord(Nb(j,1),2) Coord(Nb(j,1),1)-Coord(Nb(j,2),1)]/led;
        b(Nb(j,:))= b(Nb(j,:))+led*u_N(Coord(Nb(j,1),:),Coord(Nb(j,2),:),NormalV)/2;
    end
end

FullNodes=[1:size(Coord,1)];
FreeNodes=setdiff(FullNodes,unique(Db));

uh=zeros(length(FullNodes),1);
if (~isempty(Db))
    Dbnodes=unique(Db);
    for j=1:size(Dbnodes,1)
        uh(Dbnodes(j),1)=ue(Coord(Dbnodes(j),:));
    end
end

uh(FreeNodes)=A(FreeNodes,FreeNodes)\(b(FreeNodes)-A(FreeNodes,FullNodes)*uh);

% Compute the Derivative of uh
[uxh,uyh]=estuxh(uh,Elem,Coord,grad4e);

etaE=ResidualEdge(uxh,uyh,Elem,Coord,n2ed,ed2el,Nb);
etaf=Residualf(Elem,Coord,area4e);
Estm=sqrt(sum(etaE.^2+etaf.^2));

u=u_nodes(Coord);
%H1e=sqrt((u-uh)'*A*(u-uh));
[H1e,L2e]=Err(Coord, Elem,grad4e,area4e, uh);
ct=cputime-tt;

Ndof = length(FreeNodes); IteLevel=1; 
%data=[ct,L2e,Estm];
data=[Ndof,L2e,1/Ndof,1/(Ndof^(2/3))];

% No of Max Iterations and Max number of DOF.
maxNdof=2100; maxIteLevel=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while and(Ndof <= maxNdof, IteLevel<=maxIteLevel)

  
    [Coord,Elem,Nb,Db]=bisectionedge(Coord,Elem,n2el,n2ed,Nb,Db,nrEdges,etaf+etaE,0.3);
    
%     figure(1)
%     triplot(Elem,Coord(:,1),Coord(:,2));
%     drawnow;

    area4e = getArea4e(Coord,Elem);
    grad4e = getGrad4e(Coord,Elem,area4e);
    [n2el,n2ed,ed2el]=edge(Elem,Coord);

    nrEdges=size(ed2el,1);
    nrElems=size(Elem,1);
    nrNodes=size(Coord,1);

    A=sparse(size(Coord,1),size(Coord,1));
    b=sparse(size(Coord,1),1);

    for j=1:size(Elem,1),
        area=area4e(j);
        grads=grad4e(:,:,j);
        A(Elem(j,:),Elem(j,:))=A(Elem(j,:),Elem(j,:))+stima(area,grads);
    end

    for j=1:size(Elem,1),
        b(Elem(j,:))=b(Elem(j,:))+area4e(j)*f(sum(Coord(Elem(j,:),:))/3)/3;
    end

    if (~isempty(Nb))
       for j=1:size(Nb,1)
           led=norm(Coord(Nb(j,1),:)-Coord(Nb(j,2),:));
           NormalV=[Coord(Nb(j,2),2)-Coord(Nb(j,1),2) Coord(Nb(j,1),1)-Coord(Nb(j,2),1)]/led;
           b(Nb(j,:))= b(Nb(j,:))+led*u_N(Coord(Nb(j,1),:),Coord(Nb(j,2),:),NormalV)/2;
       end
    end

    FullNodes=[1:size(Coord,1)];
    FreeNodes=setdiff(FullNodes,unique(Db));

    uh=zeros(length(FullNodes),1);
    if (~isempty(Db))
        Dbnodes=unique(Db);
        for j=1:size(Dbnodes,1)
            uh(Dbnodes(j),1)=ue(Coord(Dbnodes(j),:));
        end
    end
    
    uh(FreeNodes)=A(FreeNodes,FreeNodes)\(b(FreeNodes)-A(FreeNodes,FullNodes)*uh);


    [uxh,uyh]=estuxh(uh,Elem,Coord,grad4e);
    etaE=ResidualEdge(uxh,uyh,Elem,Coord,n2ed,ed2el,Nb);
    etaf=Residualf(Elem,Coord,area4e);
    Estm=sqrt(sum(etaE.^2+etaf.^2));
    
    u=u_nodes(Coord);
    %H1e=sqrt((u-uh)'*A*(u-uh));
    [H1e,L2e, H1_rel_err, L2_rel_err]=Err(Coord, Elem, grad4e,area4e, uh);
    ct=cputime-tt;


    IteLevel=IteLevel+1
    Ndof=length(FreeNodes)
    %data=[data;ct,L2e,Estm];
     data=[data;Ndof,L2e,1/Ndof,1/(Ndof^(2/3))];
    figure(2)
    %loglog(data(:,1),data(:,2),'r-d',data(:,1),data(:,3),'b-*')
    loglog(data(:,1),data(:,2),'r-o',data(:,1),data(:,3),'k-d',data(:,1),data(:,4),'b-+')
    hold on
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % display the computed solution
u=u_nodes(Coord);
figure(3)
show(Coord,Elem,uh,u)
figure(4)
patch('Faces',Elem,'Vertices',Coord,'Facecolor','none','Edgecolor','blue')
title('Adaptive mesh')

figure(5)
show2Dproj(Coord, Elem, uh, u)
% 
% 
% figure(6)
% show_1(Coord,Elem,uh,u)

% figure(7)
% show_2(Coord,Elem,uh,u)

H1e,L2e, H1_rel_err, L2_rel_err, nrNodes, nrElems



