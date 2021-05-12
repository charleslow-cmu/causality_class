function ind = findindex(ind_mixed, Ind_remaining, Lset, active_L)
% function ind = findindex(ind_mixed, Ind_remaining, Lset)
% to find the indices of the X variables given the mixed indices (some are 
ind = [];
for i=1:length(ind_mixed)
    if ind_mixed(i)<= length(Ind_remaining)
        ind = [ind Ind_remaining(ind_mixed(i))];
    else
        ind = [ind  Lset{ active_L( ind_mixed(i)-length(Ind_remaining) ),4} ];
    end
end
    