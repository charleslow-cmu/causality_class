% code for simulation studies of estimating linear-Gaussian hierarchical
% structure

% L1 -> L2; L1 -> L3; L1 -> L4; L1 -> L5? L1 -> L6; L6 -> L7; L6 -> L8; 
% L2 -> X1,2,3; L3,4 -> X4,5,6,7,8; L5 ->X9,10 in specific ways
D_L = 8;
N = 2000;
EE = normrnd(0,1,N,D_L);
B_L = [0 0 0 0 0 0 0 0; .5 0 0 0 0 0 0 0; .8 0 0 0 0 0 0 0; 1 0 0 0 0 0 0 0; -.7 0 0 0 0 0 0 0; .8 0 0 0 0 0 0 0; 0 0 0 0 0 .7 0 0; 0 0 0 0 0 1 0 0];
LL = EE * (inv(eye(D_L) - B_L))';

% generate measured variables
D_X = 14;
EE_X = normrnd(0,1,N,D_X);
A_X = [0 .6 0 0 0 0 0 0; 0 .8 0 0 0 0 0 0; 0 1 0 0 0 0 0 0; 0 0 .7 0 0 0 0 0; 0 0 .9 .6 0 0 0 0; 0 0 .4 1 0 0 0 0; 0 0 .8 3 0 0 0 0;...
    0 0 0 .8 0 0 0 0; 0 0 0 0 -0.7 0 0 0; 0 0 0 0 0.7 0 0 0; 0 0 0 0 0 0 .8 0; 0 0 0 0 0 0 .6 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 0 -0.7];
XX = LL * A_X' + .6 * EE_X;

Lset = estim_strc1(XX);
% plot the graph
plot_hier(Lset,size(XX,2));