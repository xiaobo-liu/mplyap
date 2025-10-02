function [Z, Y] = rank_trunc(prec, varargin)
%RANK_TRUNC  Row truncation of solution factor of Lyapunov equations.
%
%   SYNTAX:  
%
%   1.               Z = RANK_TRUNC(PREC, Z)
%   computes a row compression of the solution factor Z of 
%   A*X + X*A^T + C*C^T = 0 such that X = Z*Z^T.
%
%   2.            [Z, Y] = RANK_TRUNC(PREC, Z, Y)
%   computes a row compression of the solution factors Z and Y of 
%   A*X + X*A^T + C*S*C^T = 0 such that X = Z*Y*Z^T.
% 
%   The simulated precision is specified by PREC = 0, 1, 2, corresponding 
%   to half, single, and double precisions, respectively.

narginchk(2, 3);
if nargin == 2 
    type = sprintf('chol');
    use_ldlt = false;
    Z = varargin{1};
else
    type = sprintf('ldlt');
    Z = varargin{1};
    Y = varargin{2};
    use_ldlt = true;
end

switch prec
    case {'half'}
        u = 2^(-8);
        Z = chop(Z);
        if use_ldlt
            Y = chop(Y);
        end
    case {'single'}
        u = eps(prec)/2;
        Z = single(Z);
        if use_ldlt
            Y = single(Y);
        end
    case {'double'}
        u = eps(prec)/2;
    otherwise
        error('Specified working precision not supported.');
end

switch type
    case {'chol'}
        epsilon = sqrt(u); % tolerances for the chol-type rank truncation
        Z_normt = norm(Z, 2);
        truncate_eps = epsilon * Z_normt;
        [~, Z, Pi] = qr(Z');
        idx = abs(diag(Z)) > truncate_eps;
        Z = Pi * Z(idx, :)' ;
    case {'ldlt'}
        epsilon = u;
        [Q, R] = qr(Z, 'econ'); 
        H = R * Y * R';
        [V, Y] = eig(H);
        truncate_eps = epsilon * max(abs(diag(Y)));
        idx = abs(diag(Y)) > truncate_eps;
        Y = Y(idx, idx);
        Z = Q * V(:, idx);
    otherwise
        error('Specified compression type not supported.');
end
end