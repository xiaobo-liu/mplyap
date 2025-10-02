% A simple test

addpath include/ external/
format compact
warning off
rng default

prec_solve = 0;
prec_work = 2;
prec_resfac = 2; 
prec_solupt = prec_work;

cond_magnitude = 2.5;

n = 100; 
ratio = 0.03;

m = n * ratio;

L = randn(n, m);
S = rand(1, m);
U = gallery('orthog', m);
S = U .* S / U;

if prec_work == 1
    L = single(L);
    S = single(S);
    L_ldlt = L;
    W_ldlt = L * S * L';  % W_ldlt = L_ldlt * S * L_ldlt'
    L_chol = L * chol(S)';
    W_chol = L_chol * L_chol';  % W_chol = L_chol * L_chol'
    W_normdiff = norm(W_ldlt-W_chol, 2) / norm(W_chol, 2)
    
    % A needs be stable - having eigenvalues with negative real part
    D = - logspace(0, cond_magnitude, n);
    D = single(D);
    V = gallery('orthog', n);
    V = single(V);
    A = V .* D / V;
else
    L_ldlt = L;
    W_ldlt = L * S * L'; 
    L_chol = L * chol(S)';
    W_chol = L_chol * L_chol'; 
    W_normdiff = norm(W_ldlt-W_chol, 2) / norm(W_chol, 2)
    D = - logspace(0, cond_magnitude, n);
    V = gallery('orthog', n);
    A = V .* D / V;
end

cond_lyap = cond(A, 1);
cond_lyap

%% from MATLAB lyap
Xlyap = lyap(A,W_chol); 
Xlyap_rank = rank(Xlyap);
Xlyap_norm = norm(Xlyap);
Xlyap_rank

%% from sign function method with Newton iteration
[num_it_chol, Zsn_chol] = lyap_sn(prec_work, A, L_chol);
Xsn_chol = Zsn_chol * Zsn_chol';
Xsn_chol_rank = rank(Xsn_chol);
Xsn_chol_rank

res_deno_sn_chol = double(2*norm(A)*norm(Xsn_chol)+norm(W_chol));
res_sn_chol = double(norm(A*Xsn_chol+Xsn_chol*A.'+W_chol)) / res_deno_sn_chol;

res_sn_chol

[num_it_ldlt, Zsn_ldlt, Ysn_ldlt] = lyap_sn(prec_work, A, L_ldlt, S);
Xsn_ldlt = Zsn_ldlt * Ysn_ldlt * Zsn_ldlt.';
Xsn_ldlt_rank = rank(Xsn_ldlt);
Xsn_ldlt_rank

res_deno_sn_ldlt = double(2*norm(A)*norm(Xsn_ldlt)+norm(W_ldlt));
res_sn_ldlt = double(norm(A*Xsn_ldlt+Xsn_ldlt*A.'+W_ldlt)) / res_deno_sn_ldlt;

res_sn_ldlt

%% from iterative refinement by sign function method with Newton iteration
[ir_step_chol, nt_iter_all_chol, nt_iter_max_chol, relres_min_chol, Z_ir_chol] = lyap_snir(prec_solve, ...
    prec_resfac, prec_solupt, A, L_chol);
Xir_chol = Z_ir_chol * Z_ir_chol.';
Xir_chol_rank = rank(Xir_chol);
Xir_chol_rank

res_deno_ir_chol = double(2*norm(A)*norm(Xir_chol)+norm(W_chol));
res_ir_chol = double(norm(A*Xir_chol+Xir_chol*A.'+W_chol)) / res_deno_ir_chol;

res_ir_chol

[ir_step_ldlt, nt_iter_all_ldlt, nt_iter_max_ldlt, relres_min_ldlt, Z_ir_ldlt, Y_ir_ldlt] = lyap_snir(prec_solve, ...
    prec_resfac, prec_solupt, A, L_ldlt, S);
Xir_ldlt = Z_ir_ldlt * Y_ir_ldlt * Z_ir_ldlt.';
Xir_ldlt_rank = rank(Xir_ldlt);
Xir_ldlt_rank

res_deno_ir_ldlt = double(2*norm(A)*norm(Xir_ldlt)+norm(W_ldlt));
res_ir_ldlt = double(norm(A*Xir_ldlt+Xir_ldlt*A.'+W_ldlt)) / res_deno_ir_ldlt;

res_ir_ldlt

%% plot singular value decay

[~, sigma_chol, ~] = svd(Xir_chol);
[~, sigma_ldlt, ~] = svd(Xir_ldlt);
semilogy(1:n, diag(sigma_chol), '-s', 1:n, diag(sigma_ldlt), '-o', 'LineWidth', 1.2);
legend('SNIR\_Chol', 'SNIR\_LDLT')
title('Singular value decay of the full solution')