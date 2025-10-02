    function [ir_step, nt_iter_all, nt_iter_max, relres_min, varargout] = ...
        lyap_snir(prec_solve, prec_resfac, prec_solupt, A, L, varargin)
%LYAP_SNIR      Sign function method Newton solver Iterative Refinement for 
% factored solution of Lyapunov equations.
%
%   SYNTAX:  
%
%   1.    [IT_STEP, NT_ITER_ALL, NT_ITER_MAX, RELRES_MIN, Z] = ...
%                   LYAP_SNIR(PREC_SOLVE, PREC_RESFAC, PREC_SOLUPT, A, L)
%   computes the solution factor Z of A*X + X*A^T + L*L^T = 0 such that 
%   X = Z*Z^T. The algorithm performs the solver in precision PREC_SOLVE,
%   the residual factorization in precision PREC_RESFAC, and solution
%   update in precision PREC_SOLUPT.
%
%   2.   [IT_STEP, NT_ITER_ALL, NT_ITER_MAX, RELRES_MIN, Z] = ...
%        LYAP_SNIR(PREC_SOLVE, PREC_RESFAC, PREC_SOLUPT, A, L, RES_DCMTOL)
%   additionally specifies the tolerance in the residual factorization to 
%   RES_DCMTOL, if isscalar(RES_DCMTOL).
%
%   3.    [IT_STEP, NT_ITER_ALL, NT_ITER_MAX, RELRES_MIN, Z, Y] = ...
%                 LYAP_SNIR(PREC_SOLVE, PREC_RESFAC, PREC_SOLUPT, A, L, S)
%   computes the solution factors Z and Y of A*X + X*A^T + L*S*L^T = 0 such 
%   that X = Z*Y*Z^T. The algorithm performs the solver in precision 
%   PREC_SOLVE, the residual factorization in precision PREC_RESFAC, and 
%   solution update in precision PREC_SOLUPT.
%
%   4.    [IT_STEP, NT_ITER_ALL, NT_ITER_MAX, RELRES_MIN, Z, Y] = ...
%     LYAP_SNIR(PREC_SOLVE, PREC_RESFAC, PREC_SOLUPT, A, L, S, RES_DCMTOL)
%   additionally specifies the tolerance in the residual factorization to 
%   RES_DCMTOL. 
%
%   RES_DCMTOL = 1e-4 by default;
%   IT_STEP retuens the number of iterative refiment steps;
%   NT_ITER_ALL returns the total number of Newton iterations;
%   NT_ITER_MAX returns the maximal number of Newton iterations in a single
%   call of the Newton iteration.
%   RELRES_MIN = NaN if no stagnation, otherwise it returns the minimal 
%   relative residual across all refinement steps.
%
%   Inputs: A asymptotically stable, S symmetric positive semidefinite.

narginchk(5, 7);
res_dectol_default = 1e-4;
switch nargin
    case 5
        type = sprintf('chol');
        res_dcmtol = res_dectol_default;
    case 6
        if (isscalar(varargin{1}) && varargin{1}<1)
            type = sprintf('chol');
            res_dcmtol = varargin{1};
        else
            type = sprintf('ldlt');
            S = varargin{1};
            res_dcmtol = res_dectol_default;
        end
    case 7
        type = sprintf('ldlt');
        S = varargin{1};
        res_dcmtol = varargin{2};
        
end

n = size(A, 1);
normA = norm(A,'fro');

max_irstep = 50; % the maximal number of refinement steps

% since prec_solfac = prec_work;
if prec_solupt == 1
    u = eps('single')/2;
    res_tol = n * u;
else 
    u = eps('double')/2;
    res_tol = n * u;
end

ir_step = 0;
relres_prev = inf;
res_reduc_tol = 0.90;
stagn_counter = 0;
switch type
    case {'chol'}
        [init_it, Z] = lyap_sn(prec_solve, A, L);
        nt_iter_all = init_it;
        nt_iter_max = init_it;
        relres_vec = ones(max_irstep, 1, class(A));
        for i = 1:max_irstep
            % residual decomposition into psd and nsd parts 
            % Res = A*Z*Z' + Z*Z'*A.' + L*L';
            [res_norm, L_plus, L_minus] = res_fac(A, prec_resfac, ...
                L, Z, res_dcmtol); 

            % check convergence in the Frobenius norm
            norm_deno = 2*normA*norm(Z'*Z,'fro') + norm(L'*L,'fro');
            relres_nxt = res_norm / norm_deno;
            relres_vec(i) = relres_nxt;

            reduc_fac = relres_nxt / relres_prev;
            % reduc_fac
            if reduc_fac>res_reduc_tol
                stagn_counter = stagn_counter + 1;
            else
                stagn_counter = 0;
            end
            cged = (relres_nxt <= res_tol); % convergence
            stagn = (stagn_counter == 2); % stagnation
            if cged || stagn, break, end 
          
            % if isinf(res_norm) || isinf(relres_nxt) keyboard; end

            relres_prev = relres_nxt;

            % solver step
            [nt_iter_sol, Z_plus, Z_minus] = lyap_sncor(prec_solve, ...
                type, A, L_plus, L_minus);
            nt_iter_all = nt_iter_all + nt_iter_sol;
            nt_iter_max = max(nt_iter_max, nt_iter_sol);
  
            % factored solution update (implicitly)
            Z = sol_upt(prec_solupt, Z, Z_plus, Z_minus);
            ir_step = ir_step + 1;
        end
        if ~cged % stagnation
            relres_min = min(relres_vec);
        else
            relres_min = NaN; %to return the relres at convergent iterate
        end
        varargout{1} = Z;

    case {'ldlt'}
        [init_it, Z, Y] = lyap_sn(prec_solve, A, L, S);
        nt_iter_all = init_it;
        nt_iter_max = init_it;
        relres_vec = ones(max_irstep, 1, class(A));
        for i = 1:max_irstep
            [res_norm, L_delta, S_delta] = ...
                res_fac(A, prec_resfac, L, S, Z, Y, res_dcmtol); 

            % check convergence in the Frobenius norm
            norm_deno = 2*normA*norm(Z*Y*Z','fro') + norm(L*S*L','fro');
            relres_nxt = res_norm / norm_deno;
            % relres_nxt

            relres_vec(i) = relres_nxt;

            reduc_fac = relres_nxt / relres_prev;
            % reduc_fac

            if reduc_fac>res_reduc_tol
                stagn_counter = stagn_counter + 1;
            else
                stagn_counter = 0;
            end

            cged = (relres_nxt <= res_tol); % convergence
            stagn = (stagn_counter == 2); % stagnation
            if cged || stagn, break, end 
            relres_prev = relres_nxt;

            % solver step
            [nt_iter_sol, Z_delta, Y_delta] = ...
                lyap_sncor(prec_solve, type, A, L_delta, S_delta);
            nt_iter_all = nt_iter_all + nt_iter_sol;
            nt_iter_max = max(nt_iter_max, nt_iter_sol);
  
            % factored solution update (implicitly)
            [Z, Y] = sol_upt(prec_solupt, Z, Y, Z_delta, Y_delta);
            ir_step = ir_step + 1;
        end
        if ~cged % stagnation
            relres_min = min(relres_vec);
        else
            relres_min = NaN; % to return the relres at convergent iterate
        end
        varargout{1} = Z;
        varargout{2} = Y;
    otherwise
        error('Specified decomposition type not supported.');
end
end