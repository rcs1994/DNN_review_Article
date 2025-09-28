function [coordinate,element,neumann,dirichlet]=InitialMesh 


coordinate=[0 0;0 -1;-1 -1;-1 0;-1 1;0 1;1 1;1 0];

element=[1 4 2;3 2 4;4 1 5;6 5 1;1 8 6;7 6 8];

neumann=[8 7;7 6;6 5;5 4;4 3;3 2];

dirichlet=[1 8;2 1];

%dirichlet=[8 7;7 6;6 5;5 4;4 3;3 2;2 1;1 8];

%neumann=[];

%%%%% Square

% coordinate=[0 0;1 0;1 1;0 1;1/2 1/2];
% 
% element=[5 1 2;5 2 3;5 3 4;5 4 1];
% 
% neumann=[];
% 
% dirichlet=[1 2;2 3;3 4;4 1];