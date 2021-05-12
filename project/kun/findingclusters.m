function  [Lset, Ind_remaining,active_L] = findingclusters(X, Ind_remaining, Lset, active_L, latent_size, alpha);
% to find next-level clusters from clusters Lset and the remaining
% variables in X with size (latent_size + 1)

% the total number of "variables" (including the remaining variables and
% the number of detected clusters
global Go_further; % to check whether the procedure should stop: if the number of "variables" is smaller than 2*latent_size + 1, it will terminate
Remove_overlap = 0;

L_Ind_remaining = length(Ind_remaining);
% Num_var = L_Ind_remaining + size(Lset,1);
Num_var = L_Ind_remaining + length(active_L);
if Num_var <= latent_size * 2 + 1
    if latent_size == 1
        Go_further = 0;
    end
    return;
end
% remaining X variable followed by latent variables

% go through all combinations of the "variables" of size cluster_size
comb = nchoosek(1:Num_var, latent_size+1);

% go through all combinations
% first represent the clusters with a graph, then merge them by finding
% maximal cliques, and finaly give the representation
stats_bk = [];
graph_current = zeros(Num_var,Num_var);
p_val_bk = ones(Num_var,Num_var);
for ii = 1:size(comb,1)
    % the remaining index
    remaining = find(~ismember(1:Num_var, comb(ii,:)));
    % If the dimensionality is too high, we may need to do dimension
    % reduction first to increase the testing power
    ind1 = findindex(comb(ii,:), Ind_remaining, Lset, active_L);
    ind2 = findindex(remaining, Ind_remaining, Lset, active_L);
    if ~Remove_overlap
        ind1 = unique(ind1);
        ind2 = unique(ind2);
    end
    [AA,BB,rr,UU,VV,Stats] = canoncorr(X(:,ind1), X(:,ind2) );
    stats_bk = [stats_bk Stats.p(latent_size+1)];
    if length(unique(ind2)) < length(ind2)
        pause;
    end
    
    if Stats.p(latent_size+1) > alpha
        % found a cluster
        % use a graph to represen the clusters in order to merge them
        % conveniently
        for jj = 1:length(comb(ii,:))-1
            for kk=jj+1:length(comb(ii,:))
                graph_current(comb(ii,jj), comb(ii,kk)) = 1;
                graph_current(comb(ii,kk), comb(ii,jj)) = 1;
                p_val_bk(comb(ii,jj), comb(ii,kk)) = Stats.p(latent_size+1);
                p_val_bk(comb(ii,kk), comb(ii,jj)) = Stats.p(latent_size+1);
            end
        end
    end
end
figure, subplot(1,2,1), plot(stats_bk); title(int2str(latent_size)),
subplot(1,2,2), hist(stats_bk,200);
pause(0.5);

% merge the clusters
[MC] = maximalCliques(graph_current,'v2');

% keep cliques of size larger than 1
MC(:,sum(MC)==1) = [];

if size(MC,2) < 1  % no cluster was found
    return;
end

% now let's guarantee that one (measured or latent) variables belongs to
% only one cluster...  So we detect those belonging to multiple clusters, find
%  the best one (according to Fisher's method), remove this variable from
%  remaining groups, and repeat the procedure
if Remove_overlap
    while max(sum(MC,2))>1
        for ii=1:Num_var
            if sum(MC(ii , :)) > 1
                % find which cluster Lset{i+L_Ind_remaining} should belong to
                % first find all clusters involving variables ii
                Ind = find(MC(ii , :)~=0);
                logp_sum_bk = 0;
                for jj = 1:length(Ind)
                    logp_sum_bk(jj) = sum(log(p_val_bk(ii, MC(:,Ind(jj))~=0 )));
                end
                [tmp, Max_jj] = max(logp_sum_bk);
                % remove the remaining cliques and only keep the Max_jj-th
                % cluster
                for jj = 1:length(Ind)
                    if jj~= Max_jj
                        MC(ii,Ind(jj)) = 0;
                    end
                end
                MC(:,sum(MC)<=1) = [];
                break;
            end
        end
    end
end

% Now find new clusters and check which latent variables have been merged
% (incorporated in to a new cluster), which will not be used in future
L_exist = size(Lset,1);
processed_measured = [];
% now give the clusters
Ind_remove = [];
for kk = 1:size(MC,2) % the number of clusters
    % for measured variables
    Ind1 = find(MC(1:L_Ind_remaining,kk));
    processed_measured = [processed_measured Ind_remaining(Ind1)];
    active_L = [active_L L_exist+kk];
    %     Lset{L_exist+kk, 1} = latent_size;
    Lset{L_exist+kk, 2} = Ind_remaining(Ind1);
    if isempty(find(MC(1+L_Ind_remaining:end, kk)))
        Lset{L_exist+kk, 3} = [];
        Lset{L_exist+kk, 4} = Lset{L_exist+kk, 2};
    else
        Ind2 = find(MC(1+L_Ind_remaining:end, kk));
        Ind_remove = [Ind_remove active_L(Ind2)];
        Lset{L_exist+kk, 3} = active_L(Ind2);
        % constructt Lset{L_exist+kk, 4}
        Lset{L_exist+kk, 4} = Lset{L_exist+kk, 2};
        for ll = 1:length(Ind2)
            Lset{L_exist+kk, 4} = [Lset{L_exist+kk, 4} Lset{ active_L(Ind2(ll)), 4}];
        end
        Lset{L_exist+kk, 4} = unique(Lset{L_exist+kk, 4});
    end
    Lset{L_exist+kk, 1} = min(latent_size,length(Lset{L_exist+kk, 2}) + length(Lset{L_exist+kk, 3}) - 1);
end
active_L(ismember(active_L, Ind_remove )) = []; % remove the latent variables for which new latent variables are already found

% update Ind_remaining
% Ind_remaining = find(~ismember(Ind_remaining, processed_measured));
Ind_remaining(ismember(Ind_remaining, processed_measured )) = [];