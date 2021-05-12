function plot_hier_simple(Lset, D_X)

% number of latent variables and the edges
Num_L = size(Lset,1);
index_latent = [];
Name_nodes = [];
% for ii = 1:size(Lset,1)
%     index_latent{ii} = [Num_L+1 : Num_L+Lset{ii,1}];
%     Num_L = Num_L + Lset{ii,1};
% end

for ii = 1:Num_L
    Name_nodes{ii} = ['L' int2str(ii) '(' int2str(Lset{ii,1}) ')'];
end
for ii = 1:D_X
    Name_nodes{ii+Num_L} = ['X' int2str(ii)];
end

% Use an alternative representation: if Lset{i} has 2 latent variables,
% then we just consider two latent variables

N_edges = 1;
ss = [];
tt = [];
% now find the edges
for ii = 1:size(Lset,1)  % use index_latent{ii}
    if length(Lset{ii,3}) > 0 % having latent variables as causes
        for kk = 1:length(Lset{ii,3}) % connecting the latent variables
            ss(N_edges) = ii;
            tt(N_edges) = Lset{ii,3}(kk);
            N_edges = N_edges + 1;
        end
    end
    for kk = 1:length(Lset{ii,2}) % connecting the latent variables
        ss(N_edges) = ii;
        tt(N_edges) = Num_L + Lset{ii,2}(kk);
        N_edges = N_edges + 1;
    end
end

G = digraph(ss,tt);
figure, plot(G,'Layout','layered','NodeLabel',Name_nodes);
p.Marker = 's';
p.NodeColor = 'r';

%% s = [1 1 1 2 2 3 3 4 4 5 6 7];
%% t = [2 3 4 5 6 5 7 6 7 8 8 8];
%% G = digraph(s,t)
%% eLabels = {'x' 'y' 'z' 'y' 'z' 'x' 'z' 'x' 'y' 'z' 'y' 'x'};
%% nLabels = {'{0}','{x}','{y}','{z}','{x,y}','{x,z}','{y,z}','{x,y,z}'};
%% plot(G,'Layout','force','EdgeLabel',eLabels,'NodeLabel',nLabels)