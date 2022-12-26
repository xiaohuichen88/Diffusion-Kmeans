% Simulation study for comparing: 
% 1. diffusion K-means: vanilla and local scaling versions
% 2. adaptive diffusion K-means: vanilla and local scaling versions
% 3. spectral clusterings: unnormalized, random walk normalized, symmetric
% normalized.
% 
% Version: Feb/20/2019

clear all;
close all;

dry_run = false;    % dry_run = true: local computer; dry_run = false: cluster run

if ~dry_run
    % Change working directory to SDPNAL+ (good for xhchen's account: should be customized)
    oldFolder = cd('../SDPNAL+/');
    startup;

    % Change back to the working directory of diffusion K-means
    cd(oldFolder);
end

seed = round(prod(clock))+feature('GetPid');
rng(seed);

% Simulation setup
nSim = 2; % number of simulations per job
n = 2^8*3;  % number of samples
k = 3;      % number of clusters
DGP = 1;    % 1 = two annulus + 1 core; 2 = uniform on unbalanced rectangles (pure geometry); 3 = unbalanced Gaussians
local_scaling = false;    % self-tuned bandwidth by local scaling
rounding_alg = 2;       % rounding algorithm to get clustering labels: rounding_alg = 1: Chen and Yang (2018+)'s rounding algorithm; rounding_alg = 2: spectral decomposition rounding algorithm

if dry_run
    plot_figs = true;
    save_results = false;
else
    plot_figs = false;
    save_results = true;
end

% Bandwidth and random walk steps choice
if local_scaling
    lazy = 0; %0.01; % lazy random walk parameter to make the kernel psd in case of local scaling
    switch DGP
        case 1
            epsilon = 2;  % power in the diffusion distance
            % penalization parameters in the adaptive diffusion K-means
            lambda_max = 1/8/n;
            lambda_min = 1/2000/n;
        case 2
            epsilon = 1; 
            lambda_max = 1/8/n;
            lambda_min = 1/2000/n;
        case 3
            epsilon = 1.2; 
            lambda_max = 1/1000/n;
            lambda_min = 1/200000/n;
    end
else
    lazy = 0; % always equal 0 if NO local scaling (since kernel is already psd)
    epsilon = 1.2; % power in the diffusion distance: 1.2 seems quite robust to all settings... 
    % penalization parameters in the adaptive diffusion K-means
    lambda_max = 1/8/n;
    lambda_min = 1/2000/n;
    % Manually tuned (non-adaptive) kernel bandwidth parameter (without
    % local scaling) gamma: DGP=1 use 0.25; DGP=2 use 0.5; DGP=3 use ??
    switch DGP
        case 1
            gamma = 0.25; % OK! Compare with the histogram of the self tuned version: hist(diag(Sigma).^(-1),20)
        case 2
            gamma = 0.5; % OK! Compare with the histogram of the self tuned version: hist(diag(Sigma).^(-1),20)
        case 3
            % still to be tuned (values below do not work well)
            %lambda_max = 1/1000/n;
            %lambda_min = 1/200000/n;
            gamma = 1; % Compare with the histogram of the self tuned version: hist(diag(Sigma).^(-1),20): no gamma empirically works!
    end
end

L_lambda = 50; 
lambda_vec = exp(log(lambda_min):(log(lambda_max)-log(lambda_min))/(L_lambda-1):log(lambda_max));  % even spacing on log-scale
%lambda_vec = lambda_min:(lambda_max-lambda_min)/(L_lambda-1):lambda_max;           % even spacing on original scale

T = round(n^epsilon); % diffusion steps
lambda = 1/500/n;         % penalization parameter in the adaptive diffusion K-means
class_labels = perms(1:k);
tol = 0.01;     % tolarence level of trace norm selection as a constant

% Outputs
% Estimation errors for diffusion K-means clustering methods
sdp_error = zeros(nSim,1);
sdp_adaptive_error = zeros(nSim,1);
% K_best = zeros(nSim,1);
tr_X_best = zeros(nSim,1);

% Classification errors for diffusion K-means and spectral clustering methods
class_DKmeans_error = zeros(nSim,1);
class_DKmeans_adaptive_error = zeros(nSim,1);
class_spectral_unormalized_error = zeros(nSim,1);
class_spectral_rw_error = zeros(nSim,1);
class_spectral_symm_error = zeros(nSim,1);

% Begin the simulation loop
for ii=1:nSim 
    fprintf('Simulation %d\n', ii);
    
    % Make data
    switch DGP
        case 1  % uniform distribution over two annuli and one disk
            n_vec = [n/4, n/4, n/2];    % cluster sizes
            %n_vec = [round(0.02*n), round(0.2*n), n-round(0.02*n)-round(0.2*n)];    % cluster sizes
            R = [1 * rand(n_vec(1), 1); 2.5 * ones(n_vec(2), 1); 4 * ones(n_vec(3), 1)];
            %R = [1 * rand(n_vec(1), 1); 2 * ones(n_vec(2), 1); 5 * ones(n_vec(3), 1)];
            theta = 2 * pi * rand(n, 1);
            Data = [R .* cos(theta)  R .* sin(theta)] + 0.02 * randn(n, 2);
        case 2  % uniform distribution on three rectangles
            n_vec = [n-2*round(n*25/393), round(n*25/393), round(n*25/393)];
            Data = [[unifrnd(-15,8,n_vec(1),1), unifrnd(-8,8,n_vec(1),1)]; [unifrnd(10,15,n_vec(2),1), unifrnd(3,8,n_vec(2),1)]; [unifrnd(10,15,n_vec(3),1), unifrnd(-8,-3,n_vec(3),1)]];  
        case 3  % three Gaussian distributions
            alpha = [1/3, 1/3, 1/3];
            %alpha = [0.9, 0.05, 0.05];
            %alpha = [0.8, 0.1, 0.1];    % densities of the clusters are similar
            n_vec = mnrnd(n,alpha);
            %Data = [mvnrnd([-6,0],2^2*eye(2),n_vec(1)); mvnrnd([0,0],0.5^2*eye(2),n_vec(2)); mvnrnd([2,0],0.5^2*eye(2),n_vec(3))]; 
            Data = [mvnrnd([-6,0],2^2*eye(2),n_vec(1)); mvnrnd([0,0],0.5^2*eye(2),n_vec(2)); mvnrnd([2.5,0],0.5^2*eye(2),n_vec(3))]; 
            %Data = [mvnrnd([-30,0],2^2*eye(2),n_vec(1)); mvnrnd([0,0],0.5^2*eye(2),n_vec(2)); mvnrnd([30,0],0.5^2*eye(2),n_vec(3))]; 
    end
    Data = Data';

    % Plot data
    if plot_figs
        fig_flag = 1; 
        figure(fig_flag);
        hold off
        scatter(Data(1,:), Data(2,:), '.');
        fig_flag = fig_flag +1; 
    end

    % Construct the affinity matrix with Gaussian kernel
    if local_scaling
        DmD = zeros(n,n);
        for i=1:n
             for j=1:n
                  DmD(i,j) = norm(Data(:,i)-Data(:,j));
             end
        end
        B = sort(DmD);
        % This local scaling is similar to Zelnik-Manor & Perona's paper: not positive semidefinite
        Sigma = diag(1./mean(B(2:(floor(log(n))+1),:),1)); 
        D = exp(-Sigma * DmD.^2 * Sigma/2);
    else
        D = zeros(n,n);
        for i=1:n
            for j=1:n
                 D(i,j) = exp(-norm(Data(:,i)-Data(:,j))^2/2/gamma^2);
            end
        end
    end

    %%%%%% Begin: SDP diffusion K-means (with known K)
    X = D_kmeans_sdp(Data, k, D, T, lazy);

    % Plot of SDP solution
    if plot_figs
        figure(fig_flag)
        hold off
        imagesc(X)
        title('SDP solution (known K)')
        colormap('gray')
        colormap(flipud(colormap));
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        axis equal
        a=axis;
        a(1)=a(3);
        a(2)=a(4);
        axis(a)
        fig_flag = fig_flag +1; 
    end

    % Check the SDP error
    % true membership labels
    X_true_label = 1 ./ n_vec;
    X_true = blkdiag(X_true_label(1)*ones(n_vec(1)), X_true_label(2)*ones(n_vec(2)), X_true_label(3)*ones(n_vec(3)));
    
    % True membership matrix
    if plot_figs
        figure(fig_flag)
        hold off
        imagesc(X_true)
        title('True membership matrix')
        colormap('gray')
        colormap(flipud(colormap));
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        axis equal
        a=axis;
        a(1)=a(3);
        a(2)=a(4);
        axis(a)
        fig_flag = fig_flag +1; 
    end

    % Absolute L_1 estimation error (normalized by n)
    sdp_error(ii) = sum(sum(abs(X-X_true)))/n;

    % Rounding to get the cluster labels
    if 1 == rounding_alg
        IDX = rounding(X);  % Chen and Yang's rounding algorithm: this rounding algorithm can be improved
    elseif 2 == rounding_alg
        % Alternative "rounding" by spectral properties of the SDP solution X
        [eigVectors,eigValues] = eig(X);
        [eigValues_X_sorted, eval_idx] = sort(diag(eigValues),'descend');
        eigVectors_sorted = eigVectors(:,eval_idx);

        % select k largest eigen vectors
        U = eigVectors_sorted(:,1:k);

        % perform kmeans clustering on the matrix U
        [IDX,C] = kmeans(U,k); 
    end
    
    % Plot the data with clustering labels
    if plot_figs
        % unique(IDX)
        figure(fig_flag)
        hold on;
        for i=1:size(IDX,1)
            if IDX(i,1) == 1
                plot(Data(1,i),Data(2,i),'m+');
            elseif IDX(i,1) == 2
                plot(Data(1,i),Data(2,i),'g+');
            else
                plot(Data(1,i),Data(2,i),'b+');        
            end
        end
        hold off;
        title('SDP Diffusion K-means (known K)');
        grid on;shg
        fig_flag = fig_flag +1; 
    end

    % Check classification error (normalized by n)
    IDX_true = zeros(n,factorial(k));  % contains all possible permutations
    class_error_all = zeros(1,factorial(k));
    for i=1:factorial(k)
        IDX_true(1:n_vec(1),i) = class_labels(i,1);
        IDX_true((n_vec(1)+1):(n_vec(1)+n_vec(2)),i) = class_labels(i,2);
        IDX_true((n_vec(1)+n_vec(2)+1):end,i) = class_labels(i,3);
        class_error_all(i) = sum(IDX~=IDX_true(:,i));
    end
    class_DKmeans_error(ii) = min(class_error_all)/n;
    %%%%%% End: SDP diffusion K-means (with known K)

    %%%%%% Begin: SDP diffusion K-means (with unknown K, i.e., adaptive)
    X_adaptive = zeros(n,n,L_lambda);
    sdp_adaptive_error_initial = zeros(1,L_lambda);
%     IDX = zeros(n,L_lambda);
    tr_X = zeros(L_lambda,1);
%     k_hat = zeros(L_lambda,1);

    for i_lambda=1:L_lambda
        lambda_i = lambda_vec(i_lambda);
        X_adaptive(:,:,i_lambda) = D_kmeans_sdp_adaptive(Data, lambda_i, D, T, lazy);

        % Rounding to get the cluster labels (better rounding algorithms?)
%         IDX(:,i_lambda) = rounding(X_adaptive(:,:,i_lambda));
        
        % Calculating the trace of the SDP solution
        tr_X(i_lambda) = trace(X_adaptive(:,:,i_lambda));

        % Absolute L_1 estimation error (normalized by n)
        sdp_adaptive_error_initial(i_lambda) = sum(sum(abs(X_adaptive(:,:,i_lambda)-X_true)))/n;

        % Number of estimated clusters
%         k_hat(i_lambda) = max(IDX(:,i_lambda));     % Check: should we use unique instead??
    end

    % Plot of k_hat versus lambda
    if plot_figs
        figure(fig_flag)
        %plot(log(lambda_vec), k_hat,'*-');
        plot(log(lambda_vec), tr_X,'*-');
        xlabel('$\log(\lambda)$','Interpreter','latex');
        ylabel('$\mbox{trace}(\tilde{Z})$','Interpreter','latex');
        %plot(n*lambda_vec, k_hat,'*-');
        %xlabel('n \times \lambda');
        %ylabel('Number of clusters');
        fig_flag = fig_flag + 1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%% Select the best K (not stable due to rounding)
%     k_hat_unique = unique(k_hat);
%     k_hat_L = length(k_hat_unique);
%     
%     lambda_length = zeros(k_hat_L,1);
%     
%     for i_k_hat = 1:k_hat_L
%         k_hat_tmp = k_hat_unique(i_k_hat);
%         tmp = find(k_hat_tmp==k_hat);
%         lambda_length(i_k_hat) = range(log(lambda_vec(tmp)));
%     end
%     K_best(ii) = k_hat_unique(find(max(lambda_length)==lambda_length));
%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Select the best K without rounding (using (tr(X_adaptive)))
%     if range(tr_X) <= 0.2
%         tr_X_best(ii) = round(mean(tr_X));
%     else
%         delta_tr_X = diff(tr_X);
%         jp_tr_X = find(abs(delta_tr_X) > 0.2);      % jump point of tr(X_adaptive) along lambda
%         lambda_flat_length = 0;
%         if range(tr_X(1:jp_tr_X(1))) <= 0.2
%             lambda_flat_length = range(log(lambda_vec(1:jp_tr_X(1))));
%             tr_X_best(ii) = round(tr_X(1));
%         end
% 
%         for i_jp = 2:length(jp_tr_X)
%             if range(tr_X((jp_tr_X(i_jp-1)+1):jp_tr_X(i_jp))) <= 0.2 & range(log(lambda_vec(jp_tr_X(i_jp-1):jp_tr_X(i_jp)))) > lambda_flat_length
%                 tr_X_best(ii) = round(tr_X(jp_tr_X(i_jp)));
%                 lambda_flat_length = range(log(lambda_vec(jp_tr_X(i_jp-1):jp_tr_X(i_jp))));
%             end
%         end
% 
%         if range(tr_X((jp_tr_X(i_jp)+1):end)) <= 0.2 & range(log(lambda_vec((jp_tr_X(i_jp)+1):end))) > lambda_flat_length
%             tr_X_best(ii) = round(tr_X(end));
%             lambda_flat_length = range(log(lambda_vec((jp_tr_X(i_jp)+1):end)));
%         end
%     end
    
    % Modified trace norm criterion: better than above
    if range(tr_X) <= tol
        tr_X_best(ii) = round(mean(tr_X));
    else
        delta_tr_X = diff(tr_X);
        tmp = (abs(delta_tr_X) <= tol);
        longest=max(accumarray(nonzeros((cumsum(~tmp)+1).*tmp),1));
        longest_pattern = ones(longest,1);
        longest_begin_idx = strfind(tmp',longest_pattern');
        longest_end_idx = longest_begin_idx+longest-1;
        tr_X_best(ii) = round(mean(tr_X(longest_begin_idx:(longest_end_idx+1))));
    end
    
    % Rerun the SDP diffusion K-means with selected K
    X_adaptive_rerun = D_kmeans_sdp(Data, tr_X_best(ii), D, T, lazy);
    
    % Absolute L_1 estimation error (normalized by n)
    sdp_adaptive_error(ii) = sum(sum(abs(X_adaptive_rerun-X_true)))/n;
    
    % Plot of SDP solution
    if plot_figs
        figure(fig_flag)
        hold off
        imagesc(X_adaptive_rerun)
        title('SDP solution adaptive')
        colormap('gray')
        colormap(flipud(colormap));
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        axis equal
        a=axis;
        a(1)=a(3);
        a(2)=a(4);
        axis(a)
        fig_flag = fig_flag +1; 
    end
    
    % Rounding to get the cluster labels
    if 1 == rounding_alg
        IDX = rounding(X_adaptive_rerun);  % this rounding algorithm can be improved
    elseif 2 == rounding_alg
        % Alternative "rounding" by spectral properties of the SDP solution X
        [eigVectors,eigValues] = eig(X_adaptive_rerun);
        [eigValues_X_sorted, eval_idx] = sort(diag(eigValues),'descend');
        eigVectors_sorted = eigVectors(:,eval_idx);

        % select k largest eigen vectors
        U = eigVectors_sorted(:,1:tr_X_best(ii));

        % perform kmeans clustering on the matrix U
        [IDX,C] = kmeans(U,tr_X_best(ii)); 
    end

    % Plot the data with clustering labels
    if plot_figs
        figure(fig_flag)
        hold on;
        for i=1:size(IDX,1)
            if IDX(i,1) == 1
                plot(Data(1,i),Data(2,i),'m+');
            elseif IDX(i,1) == 2
                plot(Data(1,i),Data(2,i),'g+');
            else
                plot(Data(1,i),Data(2,i),'b+');        
            end
        end
        hold off;
        title('SDP Diffusion K-means adaptive');
        grid on;shg
        fig_flag = fig_flag +1; 
    end
    
    % Check classification error (normalized by n)
    class_adaptive_error_all = zeros(1,factorial(k));
    for i=1:factorial(k)
        class_adaptive_error_all(i) = sum(IDX~=IDX_true(:,i));
    end
    class_DKmeans_adaptive_error(ii) = min(class_adaptive_error_all)/n;
    %%%%%% End: SDP diffusion K-means (with unknown K, i.e., adaptive)


    %%%%%%%%%%%% Spectral clustering: three methods
    DD = diag(sum(D,2));  % degree matrix

    %%%%%% Begin: Unnormalized graph Laplacian
    L = DD - D; % unnormalized graph Laplacian matrix
    [eigVectors,eigValues] = eig(L);
    [eigValues_L_sorted, eval_idx] = sort(diag(eigValues));
    eigVectors_sorted = eigVectors(:,eval_idx);

    % select k smallest eigen vectors
    U = eigVectors_sorted(:,1:k);

    % perform kmeans clustering on the matrix U
    [IDX,C] = kmeans(U,k); 

    % Plot the data with clustering labels
    if plot_figs
        figure(fig_flag)
        hold on;
        for i=1:size(IDX,1)
            if IDX(i,1) == 1
                plot(Data(1,i),Data(2,i),'m+');
            elseif IDX(i,1) == 2
                plot(Data(1,i),Data(2,i),'g+');
            else
                plot(Data(1,i),Data(2,i),'b+');        
            end
        end
        hold off;
        title('Spectral clustering: unnormalized Laplacian');
        grid on;shg
        fig_flag = fig_flag +1; 
    end
    
    % Check classification error (normalized by n)
    class_spectral_unormalized_error_all = zeros(1,factorial(k));
    for i=1:factorial(k)
        class_spectral_unormalized_error_all(i) = sum(IDX~=IDX_true(:,i));
    end
    class_spectral_unormalized_error(ii) = min(class_spectral_unormalized_error_all)/n;
    %%%%%% End: Unnormalized graph Laplacian


    %%%%%% Begin: Normalized Laplacian: random walk
    NL_RW = eye(n) - DD \ D;   % I - DD^-1 * D = DD^{-1} * (DD - D)
    [eigVectors,eigValues] = eig(NL_RW);
    [eigValues_NL_RW_sorted, eval_idx] = sort(diag(eigValues));
    eigVectors_sorted = eigVectors(:,eval_idx);

    % select k smallest eigen vectors
    U_RW = eigVectors_sorted(:,1:k);

    % perform kmeans clustering on the matrix U
    [IDX,C] = kmeans(U_RW,k); 

    % Plot the data with clustering labels
    if plot_figs
        figure(fig_flag)
        hold on;
        for i=1:size(IDX,1)
            if IDX(i,1) == 1
                plot(Data(1,i),Data(2,i),'m+');
            elseif IDX(i,1) == 2
                plot(Data(1,i),Data(2,i),'g+');
            else
                plot(Data(1,i),Data(2,i),'b+');        
            end
        end
        hold off;
        title('Spectral clustering: random walk Laplacian');
        grid on;shg
        fig_flag = fig_flag +1; 
    end
    % Check classification error (normalized by n)
    class_spectral_rw_error_all = zeros(1,factorial(k));
    for i=1:factorial(k)
        class_spectral_rw_error_all(i) = sum(IDX~=IDX_true(:,i));
    end
    class_spectral_rw_error(ii) = min(class_spectral_rw_error_all)/n;
    %%%%%% End: Normalized Laplacian: random walk


    %%%%%% Begin: Normalized (symmetric) Laplacian: Ng, Jordan, Weiss 
    NL_NJW = eye(n) - DD^(-1/2) * D * DD^(-1/2);
    [eigVectors,eigValues] = eig(NL_NJW);
    [eigValues_NL_NJW_sorted, eval_idx] = sort(diag(eigValues));
    eigVectors_sorted = eigVectors(:,eval_idx);

    % select k smallest eigen vectors
    nEigVec = eigVectors_sorted(:,1:k);

    % construct the normalized matrix U from the obtained eigen vectors
    U_NJW = zeros(n,k);
    for i=1:n
        Norm = sqrt(sum(nEigVec(i,:).^2));    
        U_NJW(i,:) = nEigVec(i,:) ./ Norm; 
    end

    % perform kmeans clustering on the matrix U
    [IDX,C] = kmeans(U_NJW,k); 

    % Plot the data with clustering labels
    if plot_figs
        figure(fig_flag)
        hold on;
        for i=1:size(IDX,1)
            if IDX(i,1) == 1
                plot(Data(1,i),Data(2,i),'m+');
            elseif IDX(i,1) == 2
                plot(Data(1,i),Data(2,i),'g+');
            else
                plot(Data(1,i),Data(2,i),'b+');        
            end
        end
        hold off;
        title('Spectral clustering: symmetrized Laplacian');
        grid on;shg
        fig_flag = fig_flag +1; 
    end
    % Check classification error (normalized by n)
    class_spectral_symm_error_all = zeros(1,factorial(k));
    for i=1:factorial(k)
        class_spectral_symm_error_all(i) = sum(IDX~=IDX_true(:,i));
    end
    class_spectral_symm_error(ii) = min(class_spectral_symm_error_all)/n;
    %%%%%% End: Normalized Laplacian: Ng, Jordan, Weiss 

    %%%%%% Plot the eigenvalues and eigenvectors
    if plot_figs
        deg_min = min(diag(DD));        % minimum degree of the graph

        figure(fig_flag)
        title('Plots of first three eigenvectors');
        subplot(3,4,1)
        plot(1:10,eigValues_L_sorted(1:10),'*')
        subplot(3,4,2)
        plot(1:n,U(:,1))
        ytickformat('%.3f')
        subplot(3,4,3)
        plot(1:n,U(:,2))
        ytickformat('%.3f')
        subplot(3,4,4)
        plot(1:n,U(:,3))
        ytickformat('%.3f')

        subplot(3,4,5)
        plot(1:10,eigValues_NL_RW_sorted(1:10),'*')
        subplot(3,4,6)
        plot(1:n,U_RW(:,1))
        ytickformat('%.3f')
        subplot(3,4,7)
        plot(1:n,U_RW(:,2))
        ytickformat('%.3f')
        subplot(3,4,8)
        plot(1:n,U_RW(:,3))
        ytickformat('%.3f')

        subplot(3,4,9)
        plot(1:10,eigValues_NL_NJW_sorted(1:10),'*')
        subplot(3,4,10)
        plot(1:n,U_NJW(:,1))
        ytickformat('%.3f')
        subplot(3,4,11)
        plot(1:n,U_NJW(:,2))
        ytickformat('%.3f')
        subplot(3,4,12)
        plot(1:n,U_NJW(:,3))
        ytickformat('%.3f')
        fig_flag = fig_flag +1; 

        %%%%%% 3D plots of the eigenvectors
        figure(fig_flag)
        plot3(U(:,1),U(:,2),U(:,3),'.');
        title('Embedding: unnormalized Laplacian');
        fig_flag = fig_flag +1; 

        figure(fig_flag)
        plot3(U_RW(:,1),U_RW(:,2),U_RW(:,3),'.');
        title('Embedding: random walk normalized Laplacian');
        fig_flag = fig_flag +1; 

        figure(fig_flag)
        plot3(U_NJW(:,1),U_NJW(:,2),U_NJW(:,3),'.');
        title('Embedding: symmetrized Laplacian');
        fig_flag = fig_flag +1; 

        %%%%%% Graph degree distribution
        figure(fig_flag)
        hist(diag(DD));
        fig_flag = fig_flag +1; 
    end
    
    fprintf('\n');
end     % End of the simulation loop

if save_results
    filename = strcat('DGP=', num2str(DGP),'_seed=', num2str(seed),'.mat');
    save(filename);
end
