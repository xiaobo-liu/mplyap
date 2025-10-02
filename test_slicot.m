% TEST_SLICOT Test the main algorithm on problems from the SLICOT library
addpath ('include', 'external', 'testmats','data')
format compact
format shorte
rng default 
warning off

dataname = sprintf('data/test_slicot.mat');

file_names = {'beam', 'build', 'CDplayer', 'eady', 'fom', 'heat-cont', 'iss', 'pde', 'random'};

num_file = length(file_names);

Acell = cell(num_file, 1);
Lcell = cell(num_file, 1);
Wcell = cell(num_file, 1);

for k = 1:num_file
    filename = ['testmats/', file_names{k}];
    filename = strcat(filename, ".mat");
    load(filename);
    Acell{k} = full(A);  
    Lcell{k} = full(C).'; % AX + XA + C^TC = 0 from the data files;
    Wcell{k} = Lcell{k}*Lcell{k}.';
end

% Main test
res_dcmtol = 1e-4;
prec_work_vec = [1 2];
prec_solve_vec = [0 1 2];
num_prec_work = length(prec_work_vec);
num_prec_solve = length(prec_solve_vec);

iter_all_chol = zeros(num_prec_work, num_prec_solve, num_file);
iter_max_chol = zeros(num_prec_work, num_prec_solve, num_file);
res_chol = zeros(num_prec_work, num_prec_solve, num_file);
solrank_chol = zeros(num_prec_work, num_prec_solve, num_file);
res_min_chol = zeros(num_prec_work, num_prec_solve, num_file);

iter_all_ldlt = zeros(num_prec_work, num_prec_solve, num_file);
iter_max_ldlt = zeros(num_prec_work, num_prec_solve, num_file);
res_ldlt = zeros(num_prec_work, num_prec_solve, num_file);
solrank_ldlt = zeros(num_prec_work, num_prec_solve, num_file);
res_min_ldlt = zeros(num_prec_work, num_prec_solve, num_file);

cond_sign = zeros(num_file,1);
cond_A = zeros(num_file,1);

main_loop = tic; % record the time consumption
for i = 1:num_prec_work
    prec_work = prec_work_vec(i);
    prec_resfac = prec_work; 
    prec_solupt = prec_work;
    fprintf('Running the test...prec_work = %1d\n', prec_work_vec(i));
    for j = 1:num_prec_solve
        prec_solve = prec_solve_vec(j);
        fprintf('Running the test...prec_solve = %1d\n', prec_solve_vec(j));
        if prec_solve>prec_work 
            break; 
        end
        for k = 1:num_file
            if prec_work==1
                A = single(Acell{k});  
                L = single(Lcell{k});
                S = eye(size(L,2), 'single');
                W = single(Wcell{k});
            else
                A = double(Acell{k});  
                L = double(Lcell{k});
                S = eye(size(L,2), 'double');
                W = double(Wcell{k});
            end
            n = size(A, 1);
            cond_A(k) = cond(A, 'fro');
            if prec_solve==2 && prec_work==2
                fprintf('Estimating cond_sign of k = %1d (n = %4d)...\n', n, k);
                B =[A W; zeros(n) -A'];
                condfun = @(A) signm_newton(A, 3);
                cond_sign(k) = funm_condest_fro(B, condfun);
                fprintf('Running the test of k = %1d (n = %4d, nzratio = %2.4f, cond_A = %1.1e, cond_sign = %1.1e)...\n', ...
                k, n, nnz(A)/n^2, cond_A(k), cond_sign(k));
            else
                fprintf('Running the test of k = %1d (n = %4d, nzratio = %2.4f, cond_A = %1.1e)...\n', ...
                k, n, nnz(A)/n^2, cond_A(k));
            end
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
save(dataname, 'file_names', 'num_file', 'num_prec_work', 'num_prec_solve', ...
    'cond_sign', 'cond_A', ...
    'iter_all_chol', 'iter_max_chol', 'solrank_chol', 'res_chol', 'res_min_chol', ...
    'iter_all_ldlt', 'iter_max_ldlt', 'solrank_ldlt', 'res_ldlt', 'res_min_ldlt');