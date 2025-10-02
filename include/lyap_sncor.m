function [num_it, varargout] = lyap_sncor(prec, type, A, varargin)
%LYAP_SNCOR    Sign function method Newton solver for factored solutions of 
% two (correction) Lyapunov equations.
%
%   SYNTAX:  
%
%   1.    [NUM_IT, Z_PLUS, Z_MINUS] = ...
%                           LYAP_SNCOR(PREC, CHOL, A, L_PLUS, L_MINUS)
%   computes the solution factors Z_PLUS and Z_MINUS of 
%           A*X + X*A^T + Z_PLUS*Z_PLUS^T = 0
%   and
%           A*X + X*A^T + Z_MINUS*Z_MINUS^T = 0
%   such that X = Z_PLUS*Z_PLUS^T and X = Z_MINUS*Z_MINUS^T, respectively.
%   The solver is performed in precision specified by PREC = 0, 1, 2, 
%   corresponding to half, single, and double precisions, respectively.
%
%   2.    [NUM_IT, Z_Delta, Y_Delta] = ...
%                           LYAP_SNCOR(PREC, LDLT, A, L_Delta, S_Delta)
%   computes the solution factors Z_Delta and Y_Delta of 
%   X_Delta = Z_Delta*Y_Delta*Z_Delta^T such that
%           A*X_Delta + X_Delta*A^T + L_Delta*S_Delta*L_Delta^T = 0.=
%   The solver is performed in precision specified by 
%   PREC = 0, 1, 2, corresponding to half, single, and double precisions, 
%   respectively.
%   
%   NUM_IT returns the number of Newton iterations carried out.
%   Inputs: TYPE = 'CHOL' or 'LDLT', the matrix A asymptotically stable, 
%   matrix Y symmetric positive semidefinite.

narginchk(5, 5);

options.format = 'b'; options.round = 1; options.subnormal = 1; % bfloat16
% options.format = 'h'; options.round = 1; options.subnormal = 1; % fp16
chop([],options)

n = size(A, 1);

switch prec
    case 0
        prec = sprintf('half');
        u = 2^(-8);
        A = chop(real(A));
        varargin{1} = chop(real(varargin{1}));
        varargin{2} = chop(real(varargin{2}));
        I = eye(n);
    case 1
        prec = sprintf('single');
        u = eps(prec)/2;
        A = single(A);
        varargin{1} = single(varargin{1});
        varargin{2} = single(varargin{2});
        I = eye(n, 'single');
    case 2
        prec = sprintf('double');
        u = eps(prec)/2;
        A = double(A);
        varargin{1} = double(varargin{1});
        varargin{2} = double(varargin{2});
        I = eye(n);   
end

scal = 1; 
scal_tol = 1e-2;
maxit = 50; % number of maximal Newton iterations
tau = 10*sqrt(n*u); % convergence tolerance
rho = 0.1; % controls the timing of truncation   
n = size(A, 1);

num_it = 0;
reldiff = inf;
stpintwo = false;
extrit = 0;

type = lower(type);
switch type
    case {'chol'}
        Z_plus = varargin{1};
        Z_minus = varargin{2};
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
            if ~isempty(Z_plus) % update can be empty after rank truncation
                Z_plus = [sqrt(mu1)*Z_plus,...
                    sqrt(mu2)*matmul(P_inv, Z_plus, prec)];
            end
            if ~isempty(Z_minus)
                Z_minus = [sqrt(mu1)*Z_minus,...
                    sqrt(mu2)*matmul(P_inv, Z_minus, prec)];
            end
            if size(Z_plus, 2) > n*rho
                Z_plus = rank_trunc(prec, Z_plus);
            end
            if size(Z_minus, 2) > n*rho
                Z_minus = rank_trunc(prec, Z_minus);
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
        Z_plus = Z_plus / sqrt(2);
        Z_minus = Z_minus / sqrt(2);
        varargout{1} = Z_plus;
        varargout{2} = Z_minus;
    case {'ldlt'}
        Z_delta = varargin{1};
        Y_delta = varargin{2};
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
            if ~isempty(Z_delta) % update can be empty after rank truncation
                Z_delta = [Z_delta, matmul(P_inv, Z_delta, prec)];
            end
            Y_delta = blkdiag(mu1*Y_delta, mu2*Y_delta); 
            if size(Z_delta, 2) > n*rho
                [Z_delta, Y_delta] = rank_trunc(prec, Z_delta, Y_delta);
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
        Y_delta = Y_delta / 2;
        varargout{1} = Z_delta;
        varargout{2} = Y_delta;
    otherwise
        error('Specified decomposition type not supported.');
end

end