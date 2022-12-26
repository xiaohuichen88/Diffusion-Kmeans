% Given a membership matrix Z, output the clustering labels V
% Rounding alogrithm proposed in Chen & Yang (2018).

function V=rounding(Z)
    n = size(Z,1);
    V = zeros(n,1);
    idx = 1;
    k = 1;
    G_c = 1:n;
    G = [];
    
    while length(G_c)>0
        I = find(Z(idx,:) > Z(idx,idx)/2);
        V(I) = k;
        G = union(G,I);
        G_c = setdiff(1:n,G);
        idx = min(G_c); % update index
        k = k+1;
    end
end
