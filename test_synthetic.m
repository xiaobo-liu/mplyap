% TEST_SYNTHETIC Test the main algorithm on synthetic (pseudorandom) problems
addpath ('include', 'external', 'data', 'figs')
format compact
rng default 
warning off

%% global parameters and test matrix generation

m = 3;
nn = [100 1000];
num_nn = length(nn);

res_dcmtol = 1e-4;
cond_min = 0.1;
cond_max = 8;
cond_magnitudes = [cond_min 0.5:0.5:3.5 4:1:cond_max];
num_cond = length(cond_magnitudes);
num_cond_small = sum(cond_magnitudes<4);
num_cond_large = num_cond - num_cond_small;

Acell = cell(num_nn, num_cond);
Lcell = cell(num_nn, 1);
Wcell = cell(num_nn, 1);

for j=1:num_nn
        L = rand(nn(j), m);
        Lcell{j} = L;
        W = L * L.';
        Wcell{j} = W;
    for k=1:num_cond
        D = - logspace(0, cond_magnitudes(k), nn(j));
        V = gallery('orthog', nn(j));
        % V = randn(nn(j));
        A = V .* D * V';
        % cond(A)
        Acell{j,k} = A;
    end
end

%% test 1: fp64 - small condition numbers
dataname = sprintf('data/test_synthetic_fp64_condsmall.mat');
fprintf('Running the 1st test...fp64 - small cond \n');

prec_work = 2;
prec_resfac = 2; 
prec_solupt = prec_work;
prec_solve_vec = [0 1 2];
num_prec_solve = length(prec_solve_vec);

iter_max_chol = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_chol = zeros(num_prec_solve, num_nn, num_cond_small);
res_chol     = zeros(num_prec_solve, num_nn, num_cond_small);
solrank_chol = zeros(num_prec_solve, num_nn, num_cond_small);
res_min_chol = zeros(num_prec_solve, num_nn, num_cond_small);

iter_max_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
res_ldlt     = zeros(num_prec_solve, num_nn, num_cond_small);
solrank_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
res_min_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);

main_loop = tic; % record the time consumption
for i = 1:num_prec_solve
    prec_solve = prec_solve_vec(i);
    fprintf('Running the test...prec_solve = %1d\n', prec_solve_vec(i));
    for j = 1:num_nn
        L = Lcell{j};
        S = eye(size(L,2));
        W = Wcell{j};
        fprintf('Running the test...n = %2d\n', nn(j));
        for k = 1:num_cond_small
            fprintf('Running the test...cond = %2.2e\n', 10^(cond_magnitudes(k)));
            A = Acell{j,k};
            [~, iter_all_chol(i,j,k), iter_max_chol(i,j,k), res_min_chol(i,j,k), ...
                Z_irchol] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, res_dcmtol);
            [~, iter_all_ldlt(i,j,k), iter_max_ldlt(i,j,k), res_min_ldlt(i,j,k), ...
                Z_irldlt, Y_irldlt] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, S, res_dcmtol);
            X_irchol = Z_irchol * Z_irchol.';
            X_irldlt = Z_irldlt * Y_irldlt * Z_irldlt.';
            solrank_chol(i,j,k) = rank(X_irchol);
            solrank_ldlt(i,j,k) = rank(X_irldlt);
            res_deno_snir_chol = double(2*norm(A,'fro')*norm(X_irchol,'fro')+norm(W,'fro'));
            res_deno_snir_ldlt = double(2*norm(A,'fro')*norm(X_irldlt,'fro')+norm(W,'fro'));
            res_chol(i,j,k) = double(norm(A*X_irchol+X_irchol*A.'+W,'fro')) / res_deno_snir_chol; 
            res_ldlt(i,j,k) = double(norm(A*X_irldlt+X_irldlt*A.'+W,'fro')) / res_deno_snir_ldlt;
        end
    end
end
fprintf('Producing the results took %.2f minutes.\n', toc(main_loop)/60);
save(dataname, 'nn', 'num_prec_solve', 'num_nn', 'num_cond', 'cond_min', ...
    'cond_max', 'cond_magnitudes', 'num_cond_small', 'num_cond_large', ...
    'iter_all_chol', 'iter_max_chol', 'solrank_chol', 'res_chol', 'res_min_chol', ...
    'iter_all_ldlt', 'iter_max_ldlt', 'solrank_ldlt', 'res_ldlt', 'res_min_ldlt');

%% test 2: fp64 - large condition numbers
dataname = sprintf('data/test_synthetic_fp64_condlarge.mat');
fprintf('Running the 2nd test...fp64 - large cond \n');

prec_work = 2;
prec_resfac = 2; 
prec_solupt = prec_work;
prec_solve_vec = [1 2];
num_prec_solve = length(prec_solve_vec);

iter_max_chol = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_chol = zeros(num_prec_solve, num_nn, num_cond_large);
res_chol = zeros(num_prec_solve, num_nn, num_cond_large);
solrank_chol = zeros(num_prec_solve, num_nn, num_cond_large);
res_min_chol = zeros(num_prec_solve, num_nn, num_cond_large);

iter_max_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_ldlt = zeros(num_prec_solve, num_nn, num_cond_large);
res_ldlt = zeros(num_prec_solve, num_nn, num_cond_large);
solrank_ldlt = zeros(num_prec_solve, num_nn, num_cond_large);
res_min_ldlt = zeros(num_prec_solve, num_nn, num_cond_large);

main_loop = tic; % record the time consumption
for i = 1:num_prec_solve
    prec_solve = prec_solve_vec(i);
    fprintf('Running the test...prec_solve = %1d\n', prec_solve_vec(i));
    for j = 1:num_nn
        L = Lcell{j};
        S = eye(size(L,2));
        W = Wcell{j};
        fprintf('Running the test...n = %2d\n', nn(j));
        for k = 1:num_cond_large
            fprintf('Running the test...cond = %2.2e\n', 10^(cond_magnitudes(num_cond_small+k)));
            A = Acell{j,num_cond_small+k};
            [~, iter_all_chol(i,j,k), iter_max_chol(i,j,k), res_min_chol(i,j,k), ...
                Z_irchol] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, res_dcmtol);
            [~, iter_all_ldlt(i,j,k), iter_max_ldlt(i,j,k), res_min_ldlt(i,j,k), ...
                Z_irldlt, Y_irldlt] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, S, res_dcmtol);
            X_irchol = Z_irchol * Z_irchol.';
            X_irldlt = Z_irldlt * Y_irldlt * Z_irldlt.';
            solrank_chol(i,j,k) = rank(X_irchol);
            solrank_ldlt(i,j,k) = rank(X_irldlt);
            res_deno_snir_chol = double(2*norm(A,'fro')*norm(X_irchol,'fro')+norm(W,'fro'));
            res_deno_snir_ldlt = double(2*norm(A,'fro')*norm(X_irldlt,'fro')+norm(W,'fro'));
            res_chol(i,j,k) = double(norm(A*X_irchol+X_irchol*A.'+W,'fro')) / res_deno_snir_chol; 
            res_ldlt(i,j,k) = double(norm(A*X_irldlt+X_irldlt*A.'+W,'fro')) / res_deno_snir_ldlt;     
        end
    end
end
fprintf('Producing the results took %.2f minutes.\n', toc(main_loop)/60);
save(dataname, 'nn', 'num_prec_solve', 'num_nn', 'num_cond', 'cond_min', ...
    'cond_max', 'cond_magnitudes', 'num_cond_small', 'num_cond_large', ...
    'iter_all_chol', 'iter_max_chol', 'solrank_chol', 'res_chol', 'res_min_chol', ...
    'iter_all_ldlt', 'iter_max_ldlt', 'solrank_ldlt', 'res_ldlt', 'res_min_ldlt');

%% test 3: fp32 - small condition numbers
dataname = sprintf('data/test_synthetic_fp32_condsmall.mat');
fprintf('Running the 3rd test...fp32 - small cond \n');

prec_work = 1;
prec_resfac = 1; 
prec_solupt = prec_work;
prec_solve_vec = [0 1];
num_prec_solve = length(prec_solve_vec);

iter_max_chol = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_chol = zeros(num_prec_solve, num_nn, num_cond_small);
res_chol = zeros(num_prec_solve, num_nn, num_cond_small);
solrank_chol = zeros(num_prec_solve, num_nn, num_cond_small);
res_min_chol = zeros(num_prec_solve, num_nn, num_cond_small);

iter_max_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
iter_all_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
res_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
solrank_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);
res_min_ldlt = zeros(num_prec_solve, num_nn, num_cond_small);

main_loop = tic; % record the time consumption
for i = 1:num_prec_solve
    prec_solve = prec_solve_vec(i);
    fprintf('Running the test...prec_solve = %1d\n', prec_solve_vec(i));
    for j = 1:num_nn
        L = single(Lcell{j}); % working precision is fp32
        S = eye(size(L,2), 'single');
        W = single(Wcell{j});
        fprintf('Running the test...n = %2d\n', nn(j));
        for k = 1:num_cond_small
            fprintf('Running the test...cond = %2.2e\n', 10^(cond_magnitudes(k)));
            A = single(Acell{j,k});
            [~, iter_all_chol(i,j,k), iter_max_chol(i,j,k), res_min_chol(i,j,k), ...
                Z_irchol] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, res_dcmtol);
            [~, iter_all_ldlt(i,j,k), iter_max_ldlt(i,j,k), res_min_ldlt(i,j,k), ...
                Z_irldlt, Y_irldlt] = lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, S, res_dcmtol);
            X_irchol = Z_irchol * Z_irchol.';
            X_irldlt = Z_irldlt * Y_irldlt * Z_irldlt.';
            solrank_chol(i,j,k) = rank(X_irchol);
            solrank_ldlt(i,j,k) = rank(X_irldlt);
            res_deno_snir_chol = double(2*norm(A,'fro')*norm(X_irchol,'fro')+norm(W,'fro'));
            res_deno_snir_ldlt = double(2*norm(A,'fro')*norm(X_irldlt,'fro')+norm(W,'fro'));
            res_chol(i,j,k) = double(norm(A*X_irchol+X_irchol*A.'+W,'fro')) / res_deno_snir_chol; 
            res_ldlt(i,j,k) = double(norm(A*X_irldlt+X_irldlt*A.'+W,'fro')) / res_deno_snir_ldlt;   
        end
    end
end
fprintf('Producing the results took %.2f minutes.\n', toc(main_loop)/60);
save(dataname, 'nn', 'num_prec_solve', 'num_nn', 'num_cond', 'cond_min', ...
    'cond_max', 'cond_magnitudes', 'num_cond_small', 'num_cond_large', ...
    'iter_all_chol', 'iter_max_chol', 'solrank_chol', 'res_chol', 'res_min_chol', ...
    'iter_all_ldlt', 'iter_max_ldlt', 'solrank_ldlt', 'res_ldlt', 'res_min_ldlt');