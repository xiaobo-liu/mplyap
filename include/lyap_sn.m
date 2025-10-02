function [num_it, varargout] = lyap_sn(prec, A, L, S)
%LYAP_SN    Sign function method Newton solver for factored solution of 
% Lyapunov equations.
%
%   SYNTAX:  
%
%   1.        [NUM_IT, Z] = LYAP_SN(PREC, A, L)
%   computes the solution factor Z of A*X + X*A^T + L*L^T = 0 such that 
%   X = Z*Z^T in simulated precision specified by PREC = 0, 1, 2, 
%   corresponding to half, single, and double precisions, respectively.
%
%   2.        [NUM_IT, Z, Y] = LYAP_SN(PREC, A, L, S)
%   computes the solution factors Z and Y of A*X + X*A^T + L*S*L^T = 0 such 
%   that X = Z*Y*Z^T in simulated precision specified by PREC = 0, 1, 2, 
%   corresponding to half, single, and double precisions, respectively.
%   
%   NUM_IT returns the number of Newton iterations carried out.
%   Inputs: A asymptotically stable, S symmetric positive semidefinite.

narginchk(3, 4);

options.format = 'b'; options.round = 1; options.subnormal = 1; % bfloat16
% options.format = 'h'; options.round = 1; options.subnormal = 1; % fp16

chop([], options)
[n, m] = size(L);

if nargin==3 
    type = sprintf('chol');
    Z = L;
    Y = eye(m);
else
    type = sprintf('ldlt');
    Z = L;
    Y = S;
end

switch prec
    case 0
        prec = sprintf('half');
        u = 2^(-8);
        A = chop(real(A));
        Z = chop(real(Z));
        Y = chop(real(Y));
        I = eye(n);
    case 1
        prec = sprintf('single');
        u = eps(prec)/2;
        A = single(A);
        Z = single(Z);
        Y = single(Y);
        I = eye(n, 'single');
    case 2
        prec = sprintf('double');
        u = eps(prec)/2;
        A = double(A);
        Z = double(Z);
        Y = double(Y);
        I = eye(n);
end

scal = 1; 
scal_tol = 1e-2;
maxit = 50; % number of maximal Newton iterations
tau = 10*sqrt(n*u); % convergence tolerance
rho = 0.1; % controls the timing of truncation   

num_it = 0;
reldiff = inf;
stpintwo = false;
extrit = 0;

switch type
    case {'chol'}
        for i=1:maxit
            P = A;
            P_inv = matinv(P, prec);
            if scal
                mu = sqrt(norm(P_inv,'fro') / norm(P,'fro'));
            else
                mu = 1;
            end
            mu1 = mu / 2;
            mu2 = 0.5 / mu;
            Z = [sqrt(mu1)*Z, sqrt(mu2)*matmul(P_inv, Z, prec)];
            if size(Z, 2) > n*rho
                Z = rank_trunc(prec, Z);
            end
            A = mu1*P + mu2*P_inv;
            num_it = num_it + 1;
            reldiff_pre = reldiff;
            reldiff = norm(A-P,'fro') / norm(A, 'fro'); 
            if scal && (reldiff < scal_tol), scal = false; end
            cged = ( normest1(A+I)<tau || reldiff > reldiff_pre/2 && ~scal );
            if cged, stpintwo = true; end
            if stpintwo, extrit = extrit + 1; end
            if extrit==2, break, end
        end 
        varargout{1} = Z / sqrt(2);

    case {'ldlt'}
        for i=1:maxit
            P = A;
            P_inv = matinv(P, prec);
            if scal
                mu = sqrt(norm(P_inv,'fro') / norm(P,'fro'));
            else
                mu = 1;
            end
            mu1 = mu / 2;
            mu2 = 0.5 / mu;
            Z = [Z, matmul(P_inv, Z, prec)];
            Y = blkdiag(mu1*Y, mu2*Y); 
            if size(Z, 2) > n*rho
                [Z, Y] = rank_trunc(prec, Z, Y);
            end
            A = mu1*P + mu2*P_inv;
            num_it = num_it + 1;
            reldiff_pre = reldiff;
            reldiff = norm(A-P,'fro') / norm(A, 'fro'); 
            if scal && (reldiff < scal_tol), scal = false; end
            cged = ( normest1(A+I)<tau || reldiff > reldiff_pre/2 && ~scal );
            if cged, stpintwo = true; end
            if stpintwo, extrit = extrit + 1; end
            if extrit==2, break, end
        end
        Y = Y / 2;
        varargout{1} = Z;
        varargout{2} = Y;
    otherwise
        error('Specified decomposition type not supported.');
end

end