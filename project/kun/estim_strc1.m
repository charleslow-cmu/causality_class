function Lset = estim_strc1(X)
% function Lset = estim_strc(X)
% function to estimate linear latent hierarchical structure from measured
% variables.
% X is #samples * #dimensions
% Lset is the structure of the sets of latent variables; Lset{i,1} gives the
% indicies of all the measured variables in the i-th cluster; Lset{i,2} gives the
% indicies of the immediate measured variables in the i-th cluster; Lset{i,3} gives the
% indicies of the immediate latent variables in the i-th cluster; Lset{i,4} gives the
% cardinality of the latent variables of the cluster

% Use preprocessing to increase the power of testing the number of  in the high-dimensional case?
Keep_subset = 1;

[N,D] = size(X);

if D<12
    Keep_subset = 0;
end

if Keep_subset == 1 %% later use
    % calcualte the correlation matrix to select a subset of the variables
    % for testing
    abs_corr = corr(X);
    abs_corr = abs_corr - diag(diag(abs_corr));
end

alpha = 0.01; % significance level for the testing in canonical correlation analysis
Ind_remaining = 1:D;
Lset = []; % structure to given the variables in each cluster...
active_L = [];

% clusters contains detailed information: clusters{level}{index}(1)
% contains the indices of latent children and clusters{level}{index}(2)
% contains the indices of measured children.
% step 1: finding atomic clusters
% here let's handle at most three latent variables
% some variables might not follow the model and will be finally ignored
% function to find clusters with 1 single latent variable

global Go_further;
Go_further = 1; % go until no futher cluster is obtained
level = 0;

while Go_further
    level = level + 1,
    Lset_old = Lset;
%     [Lset, Ind_remaining,active_L] = findingclusters(X, Ind_remaining, Lset, active_L, 3,alpha); 3,
%     [Lset, Ind_remaining,active_L] = findingclusters(X, Ind_remaining, Lset, active_L, 4,alpha); 4,
    for latent_size = 1:2
        latent_size,
        % finding clusters
        [Lset, Ind_remaining,active_L] = findingclusters(X, Ind_remaining, Lset, active_L, latent_size,alpha);
    end
    if size(Lset,1) == size(Lset_old,1) ; %& latent_size == 1
        Go_further = 0; break;
    end
end

